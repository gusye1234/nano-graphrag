import os
import logging
import numpy as np
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

GLM_API_KEY = "XXXX"
DEEPSEEK_API_KEY = "sk-XXXX"

MODEL = "deepseek-chat"


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-2"
    client = OpenAI(
        api_key=GLM_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)



WORKING_DIR = "./nano_graphrag_cache_deepseek_TEST"

def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        embedding_func=GLM_embedding,
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )


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
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        embedding_func=GLM_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    insert()
    # query()
