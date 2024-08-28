import os
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import gpt_4o_mini_complete
from nano_graphrag._storage import HNSWVectorStorage


WORKING_DIR = "./nano_graphrag_cache_using_hnsw_as_vectorDB"


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
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 100, "M": 8},
        best_model_max_async=1,
        cheap_model_max_async=1,
        best_model_func=gpt_4o_mini_complete,
        cheap_model_func=gpt_4o_mini_complete,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        vector_db_storage_cls=HNSWVectorStorage,
        vector_db_storage_cls_kwargs={"max_elements": 1000000, "ef_search": 100, "M": 8},
        best_model_max_async=1,
        cheap_model_max_async=1,
        best_model_func=gpt_4o_mini_complete,
        cheap_model_func=gpt_4o_mini_complete,
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )


if __name__ == "__main__":
    insert()
    query()
