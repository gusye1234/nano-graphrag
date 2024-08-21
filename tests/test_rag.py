import os
import shutil
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

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
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


def test_insert():
    with open("./tests/mock_data.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=local_embedding)
    rag.insert(FAKE_TEXT)


async def fake_model(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return FAKE_RESPONSE


def test_query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=fake_model,
        embedding_func=local_embedding,
    )
    result = rag.query("Dickens", param=QueryParam(mode="local"))
    assert result == FAKE_RESPONSE
