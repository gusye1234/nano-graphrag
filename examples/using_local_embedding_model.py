import sys

sys.path.append("..")
import logging
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_local_embedding_TEST"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
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
)

with open("../tests/mock_data.txt", encoding="utf-8-sig") as f:
    FAKE_TEXT = f.read()

# rag.insert(FAKE_TEXT)
print(rag.query("What the main theme of this story?", param=QueryParam(mode="local")))
