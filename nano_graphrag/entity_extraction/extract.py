from typing import Union
import pickle
import asyncio
from collections import defaultdict
import dspy
from nano_graphrag._storage import BaseGraphStorage
from nano_graphrag.base import (
    BaseGraphStorage,
    BaseVectorStorage,
    TextChunkSchema,
)
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import logger, compute_mdhash_id
from nano_graphrag.entity_extraction.module import EntityRelationshipExtractor
from nano_graphrag._op import _merge_edges_then_upsert, _merge_nodes_then_upsert


async def generate_dataset(
    chunks: dict[str, TextChunkSchema],
    filepath: str,
    save_dataset: bool = True
) -> list[dspy.Example]:
    entity_extractor = EntityRelationshipExtractor()
    ordered_chunks = list(chunks.items())

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]) -> dspy.Example:
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        prediction = await asyncio.to_thread(
            entity_extractor, input_text=content
        )
        example = dspy.Example(
            input_text=content, 
            entities=prediction.entities, 
            relationships=prediction.relationships
        ).with_inputs("input_text")
        return example

    examples = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    if save_dataset:
        with open(filepath, 'wb') as f:
            pickle.dump(examples, f)
            logger.info(f"Saved {len(examples)} examples with keys: {examples[0].keys()}")

    return examples


async def extract_entities_dspy(
    chunks: dict[str, TextChunkSchema],
    knwoledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    entity_extractor = EntityRelationshipExtractor()

    if global_config.get("use_compiled_dspy_entity_relationship", False):
        entity_extractor.load(global_config["entity_relationship_module_path"])
    
    ordered_chunks = list(chunks.items())
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        prediction = await asyncio.to_thread(
            entity_extractor, input_text=content
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
  
        for entity in prediction.entities.context:
            entity_dict = entity.dict()
            entity_dict["source_id"] = chunk_key
            maybe_nodes[entity_dict['entity_name']].append(entity_dict)
            already_entities += 1

        for relationship in prediction.relationships.context:
            relationship_dict = relationship.dict()
            relationship_dict["source_id"] = chunk_key
            maybe_edges[(relationship_dict['src_id'], relationship_dict['tgt_id'])].append(relationship_dict)
            already_relations += 1

        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    print()
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[k].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    return knwoledge_graph_inst
