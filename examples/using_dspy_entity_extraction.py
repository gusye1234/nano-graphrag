import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import numpy as np
import dspy
from sentence_transformers import SentenceTransformer
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import gpt_4o_mini_complete
from nano_graphrag._storage import HNSWVectorStorage
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag.entity_extraction.extract import extract_entities_dspy

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.DEBUG)

WORKING_DIR = "./nano_graphrag_cache_using_dspy_entity_extraction"

load_dotenv()


EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)


@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


async def deepseepk_model_if_cache(
    prompt, model: str = "deepseek-chat", system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content



def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def insert():
    from time import time

    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 200, "M": 50},
        best_model_max_async=10,
        cheap_model_max_async=10,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        embedding_func=local_embedding,
        entity_extraction_func=extract_entities_dspy
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 200, "M": 50},
        best_model_max_token_size=8196,
        cheap_model_max_token_size=8196,
        best_model_max_async=4,
        cheap_model_max_async=4,
        best_model_func=gpt_4o_mini_complete,
        cheap_model_func=gpt_4o_mini_complete,
        embedding_func=local_embedding,
        entity_extraction_func=extract_entities_dspy
        
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )


if __name__ == "__main__":
    lm = dspy.LM(
        model="deepseek/deepseek-chat", 
        model_type="chat",
        api_provider="openai",
        api_key=os.environ["DEEPSEEK_API_KEY"], 
        base_url=os.environ["DEEPSEEK_BASE_URL"], 
        temperature=1.0,
        max_tokens=8192
    )
    dspy.settings.configure(lm=lm, experimental=True)
    insert()
    query()
