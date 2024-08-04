import os
import asyncio
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, field, asdict
from functools import partial
from ._llm import gpt_4o_complete, gpt_4o_mini_complete
from ._utils import (
    limit_async_func_call,
    generate_id,
    EmbeddingFunc,
    logger,
)
from ._base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage, StorageNameSpace
from .storage import JsonKVStorage, MilvusLiteStorge, NetworkXStorage
from ._op import chunking_by_token_size, openai_embedding, extract_entities


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_need_summary_max_tokens: int = 4000

    # embedding
    embedding_func: EmbeddingFunc = openai_embedding
    embedding_batch_num: int = 16
    embedding_func_max_async: int = 8

    # LLM
    best_model_func: callable = gpt_4o_complete
    best_model_max_async: int = 8
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_async: int = 8

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = MilvusLiteStorge
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    enable_llm_cache: bool = False

    def __post_init__(self):

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )
        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )
        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.text_chunks_vdb = self.vector_db_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
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
            generate_id(prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")

        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                generate_id(prefix="chunk-"): {**dp, "full_doc_id": doc_key}
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            chunk_kv_storage_inst=self.text_chunks,
            global_config=asdict(self),
        )
        logger.info("[Entity Extraction] Done")

        # upsert to vector db
        # await self.text_chunks_vdb.insert(inserting_chunks)
        # upsert to KV
        await self.full_docs.upsert(new_docs)
        await self.text_chunks.upsert(chunks)

        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.text_chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            await cast(StorageNameSpace, storage_inst).index_done_callback()

        logger.info(
            f"Process {len(new_docs)} new docs, add {len(inserting_chunks)} new chunks"
        )

    def insert(self, string_or_strings):
        return asyncio.run(self.ainsert(string_or_strings))
