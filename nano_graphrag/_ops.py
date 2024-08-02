import numpy as np
import nest_asyncio

nest_asyncio.apply()
from openai import AsyncOpenAI
from ._utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    wrap_embedding_func_with_attrs,
)

openai_async_client = AsyncOpenAI()


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content,
                "chunk_order_index": index,
            }
        )
    return results


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:

    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def extract_entities(contents, prompt, use_llm_func):
    pass
