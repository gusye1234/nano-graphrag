import os
import logging
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import minimax_m2_7_complete, minimax_m2_5_highspeed_complete

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Set MiniMax API key: export MINIMAX_API_KEY="your-key-here"
# Available models: MiniMax-M2.7 (latest), MiniMax-M2.5, MiniMax-M2.5-highspeed (204K context)

WORKING_DIR = "./nano_graphrag_cache_minimax_TEST"


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
        # Use MiniMax-M2.7 as the main model and M2.5-highspeed as the cheap model
        best_model_func=minimax_m2_7_complete,
        cheap_model_func=minimax_m2_5_highspeed_complete,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=minimax_m2_7_complete,
        cheap_model_func=minimax_m2_5_highspeed_complete,
    )
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )


if __name__ == "__main__":
    insert()
    query()
