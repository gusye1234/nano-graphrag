import os
import asyncio
from typing import Type
from datetime import datetime
from dataclasses import dataclass, field, asdict
from .prompt import prompts
from ._llm import gpt_4o_complete, gpt_4o_mini_complete, chunking_by_token_size
from ._utils import (
    limit_async_func_call,
    generate_id,
    logger,
)
from .storage import JsonKVStorage, BaseStorage


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    chunk_token_size: int = 1024
    tiktoken_model_name: str = "gpt-4o"

    best_model_func: callable = gpt_4o_complete
    best_model_max_async: int = 8

    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_async: int = 8
    key_string_value_json_storage_cls: Type[BaseStorage] = JsonKVStorage

    def __post_init__(self):
        self.best_model_func = limit_async_func_call(
            max_size=self.best_model_max_async
        )(self.best_model_func)

        self.cheap_model_func = limit_async_func_call(
            max_size=self.cheap_model_max_async
        )(self.cheap_model_func)
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        logger.info(f"GraphRAG init done with param: {asdict(self)}")

    async def aquery(self, query: str):
        return await self.best_model_func(query)

    def query(self, query: str):
        return asyncio.run(self.aquery(query))

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        new_docs = {
            generate_id(prefix="doc-"): {"content": c} for c in string_or_strings
        }
        self.full_docs.update(new_docs)

        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                generate_id(prefix="chunk-"): {**dp, "full_doc_id": doc_key}
                for dp in chunking_by_token_size(
                    doc["content"],
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
        self.text_chunks.update(chunks)

    def insert(self, string_or_strings):
        return asyncio.run(self.ainsert(string_or_strings))


if __name__ == "__main__":
    a = GraphRAG()

    async def main():
        tasks = [a.aquery("What is the capital of China?") for _ in range(10)]
        results = await asyncio.gather(*tasks)
        print(results)

    asyncio.run(main())
