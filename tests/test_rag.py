import os
from nano_graphrag import GraphRAG

with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
    FAKE_TEXT = f.read()
WORKING_DIR = "./nano_graphrag_cache_TEST"


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def test_incr_insert():
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/milvus_lite.db")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    half_len = len(FAKE_TEXT) // 2
    rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    rag.insert(FAKE_TEXT[:half_len])

    rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    rag.insert(FAKE_TEXT[half_len:])


def test_query():
    rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    print(rag.query("Dickens", mode="local", top_k=1))
