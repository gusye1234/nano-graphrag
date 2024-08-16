import numpy as np
from nano_graphrag._utils import wrap_embedding_func_with_attrs

WORKING_DIR = "./nano_graphrag_cache_bge_m3_TEST"


# We're using Sentence Transformers to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def bge_embedding(texts: list[str]) -> np.ndarray:
    pass
