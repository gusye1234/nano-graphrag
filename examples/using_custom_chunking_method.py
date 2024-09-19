from nano_graphrag._utils import encode_string_by_tiktoken
from nano_graphrag.base import QueryParam
from nano_graphrag.graphrag import GraphRAG
from nano_graphrag._op import chunking_by_seperators


def chunking_by_token_size(
    tokens_list: list[list[int]],  # nano-graphrag may pass a batch of docs' tokens
    doc_keys: list[str],  # nano-graphrag may pass a batch of docs' key ids
    tiktoken_model,  # a titoken model
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


WORKING_DIR = "./nano_graphrag_cache_local_embedding_TEST"
rag = GraphRAG(
    working_dir=WORKING_DIR,
    chunk_func=chunking_by_seperators,
)
