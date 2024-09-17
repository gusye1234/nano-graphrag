

from nano_graphrag._utils import encode_string_by_tiktoken
from nano_graphrag.base import QueryParam
from nano_graphrag.graphrag import GraphRAG


def chunking_by_specific_separators(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o",
):
    from langchain_text_splitters  import RecursiveCharacterTextSplitter
    

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
        # length_function=lambda x: len(encode_string_by_tiktoken(x)),
        model_name=tiktoken_model,
        is_separator_regex=False,
        separators=[
            # Paragraph separators
            "\n\n",
            "\r\n\r\n",
            # Line breaks
            "\n",
            "\r\n",
            # Sentence ending punctuation
            "。",  # Chinese period
            "．",  # Full-width dot
            ".",  # English period
            "！",  # Chinese exclamation mark
            "!",  # English exclamation mark
            "？",  # Chinese question mark
            "?",  # English question mark
            # Whitespace characters
            " ",  # Space
            "\t",  # Tab
            "\u3000",  # Full-width space
            # Special characters
            "\u200b",  # Zero-width space (used in some Asian languages)
            # Final fallback
            "",
        ])
    texts = text_splitter.split_text(content)
    
    results = []
    for index, chunk_content in enumerate(texts):
        
        results.append(
            {
                # "tokens": None,
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


WORKING_DIR = "./nano_graphrag_cache_local_embedding_TEST"
rag = GraphRAG(
    working_dir=WORKING_DIR,
    chunk_func=chunking_by_specific_separators,
)

with open("../tests/mock_data.txt", encoding="utf-8-sig") as f:
    FAKE_TEXT = f.read()

# rag.insert(FAKE_TEXT)
print(rag.query("What the main theme of this story?", param=QueryParam(mode="local")))
