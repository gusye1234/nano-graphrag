import re
import asyncio
import numpy as np
from openai import AsyncOpenAI
from ._utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    wrap_embedding_func_with_attrs,
    pack_user_ass_to_openai_messages,
    is_float_regex,
    clean_str,
    logger,
)
from ._llm import gpt_4o_complete
from ._base import BaseGraphStorage, BaseKVStorage
from .prompt import PROMPTS

openai_async_client = AsyncOpenAI()

GRAPH_FIELD_SEP = "<SEP>"


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def extract_entities(
    chunks: dict[str, dict],
    knwoledge_graph_inst: BaseGraphStorage,
    chunk_kv_storage_inst: BaseKVStorage,
    use_llm_func: callable = gpt_4o_complete,
    entity_extract_max_gleaning=1,
) -> BaseGraphStorage:
    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    async def _process_single_content(chunk_dp: dict):
        content = chunk_dp[1]["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break
        # TODO clean the final_results
        chunk_dp[1]["raw_entities"] = final_result
        return final_result

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    logger.info("Extracting entities")
    await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
    await chunk_kv_storage_inst.upsert({k: v for k, v in ordered_chunks})

    for chunk_key, chunk_dp in ordered_chunks:
        string_r = chunk_dp["raw_entities"]
        records = []
        single_round_records = [
            r.strip()
            for r in string_r.split(context_base["record_delimiter"])
            if r.strip()
        ]
        for sr in single_round_records:
            records.extend(
                [
                    r.strip()
                    for r in sr.split(context_base["completion_delimiter"])
                    if r.strip()
                ]
            )
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = [
                r.strip()
                for r in record.split(context_base["tuple_delimiter"])
                if r.strip()
            ]

            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                # add this record as a node in the G
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])

                entity_source_id = chunk_key
                entity_node_data = await knwoledge_graph_inst.get_node(entity_name)
                if entity_node_data is not None:
                    entity_description = GRAPH_FIELD_SEP.join(
                        [entity_node_data.get("description", ""), entity_description]
                    )
                    if entity_source_id not in entity_node_data["source_id"]:
                        entity_source_id = GRAPH_FIELD_SEP.join(
                            [entity_node_data["source_id"], chunk_key]
                        )
                    else:
                        entity_source_id = entity_node_data["source_id"]
                    entity_type = entity_type or entity_node_data.get("entity_type", "")
                await knwoledge_graph_inst.upsert_node(
                    entity_name,
                    node_data=dict(
                        entity_type=entity_type,
                        description=entity_description,
                        source_id=entity_source_id,
                    ),
                )

            if record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                # add this record as edge
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = chunk_key
                weight = (
                    float(record_attributes[-1])
                    if is_float_regex(record_attributes[-1])
                    else 1.0
                )
                if not (await knwoledge_graph_inst.has_node(source)):
                    await knwoledge_graph_inst.upsert_node(
                        source, node_data=dict(source_id=edge_source_id)
                    )
                if not (await knwoledge_graph_inst.has_node(target)):
                    await knwoledge_graph_inst.upsert_node(
                        target, node_data=dict(source_id=edge_source_id)
                    )

                if await knwoledge_graph_inst.has_edge(source, target):
                    edge_data = await knwoledge_graph_inst.get_edge(source, target)
                    weight += edge_data["weight"]
                    edge_description = GRAPH_FIELD_SEP.join(
                        [
                            edge_data["description"],
                            edge_description,
                        ]
                    )
                    if edge_source_id not in edge_data["source_id"]:
                        edge_source_id = GRAPH_FIELD_SEP.join(
                            [
                                edge_data["source_id"],
                                edge_source_id,
                            ]
                        )
                    else:
                        edge_source_id = edge_data["source_id"]
                await knwoledge_graph_inst.upsert_edge(
                    source,
                    target,
                    edge_data=dict(
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    ),
                )
    return knwoledge_graph_inst
