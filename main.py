import os
import sys

sys.path.append("..")
import logging
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
from nano_graphrag._op import chunking_by_seperators

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "/mnt/rangehow/nano-graphrag/neu_cache"






from openai import AsyncOpenAI
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
# CUSTOM LLM
MODEL="default"
async def custom_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key="EMPTY", base_url="http://152.136.16.221:8203/v1"
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
        model=MODEL, messages=messages, temperature=0,**kwargs,
        timeout=10e6,
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
        await hashing_kv.index_done_callback()
    # -----------------------------------------------------
    return response.choices[0].message.content








# CUSTOM EMBEDDING

EMBED_MODEL = SentenceTransformer(
    "/mnt/rangehow/models/Conan-embedding-v1", cache_folder=WORKING_DIR, device="cpu"
)


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


rag = GraphRAG(
    working_dir=WORKING_DIR,
    embedding_func=local_embedding,
    enable_llm_cache=True,
    best_model_func=custom_model_if_cache,
    cheap_model_func=custom_model_if_cache,
    chunk_func=chunking_by_seperators,
    best_model_max_async=1024,
    cheap_model_max_async=1024,
    entity_extract_max_gleaning=0,
)

documents=[]
input_directory="/mnt/rangehow/neuspider/document/markdown_saved"
filenames = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
for filename in filenames:
    with open(os.path.join(input_directory,filename), encoding="utf-8") as f:
        string=f.read()
        if len(string)<50:
            continue
        documents.append(string)

print(len(documents))

rag.insert(documents)
print(rag.query("东北大学谁最牛逼？", param=QueryParam(mode="global")))