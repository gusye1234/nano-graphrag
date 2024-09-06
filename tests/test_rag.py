import os
import json
import shutil
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

os.environ["OPENAI_API_KEY"] = "FAKE"

WORKING_DIR = "./tests/nano_graphrag_cache_TEST"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
else:
    shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

shutil.copy(
    "./tests/fixtures/mock_cache.json",
    os.path.join(WORKING_DIR, "kv_store_llm_response_cache.json"),
)
FAKE_RESPONSE = "Hello world"
FAKE_JSON = json.dumps({"points": [{"description": "Hello world", "score": 1}]})


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


# We're using random embedding function for testing
@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


def test_insert():
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    rag = GraphRAG(
        working_dir=WORKING_DIR, embedding_func=local_embedding, enable_naive_rag=True
    )
    rag.insert(FAKE_TEXT)


async def fake_model(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return FAKE_RESPONSE


def test_local_query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_model,
        embedding_func=local_embedding,
    )
    result = rag.query("Dickens", param=QueryParam(mode="local"))
    assert result == FAKE_RESPONSE


async def fake_json_model(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return FAKE_JSON


def test_global_query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_json_model,
        embedding_func=local_embedding,
    )
    result = rag.query("Dickens")
    assert result == FAKE_JSON


def test_naive_query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_model,
        embedding_func=local_embedding,
        enable_naive_rag=True,
    )
    result = rag.query("Dickens", param=QueryParam(mode="naive"))
    assert result == FAKE_RESPONSE


def test_subcommunity_insert():
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()
    remove_if_exist(f"{WORKING_DIR}/milvus_lite.db")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=local_embedding,
        enable_naive_rag=True,
        addon_params={"force_to_use_sub_communities": True},
    )
    rag.insert(FAKE_TEXT)
