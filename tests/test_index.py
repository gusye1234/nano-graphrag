import os
import shutil
from nano_graphrag import GraphRAG

with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
    FAKE_TEXT = f.read()
WORKING_DIR = "./nano_graphrag_cache_TEST"

if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)


def test_init():
    rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    rag.insert(FAKE_TEXT)
