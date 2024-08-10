import os
import asyncio
from typing import Type, cast
from datetime import datetime
from dataclasses import dataclass, field, asdict
from functools import partial
from ._llm import gpt_4o_complete, gpt_4o_mini_complete, openai_embedding
from ._utils import (
    limit_async_func_call,
    generate_id,
    compute_mdhash_id,
    EmbeddingFunc,
    logger,
)
from ._storage import JsonKVStorage, MilvusLiteStorge, NetworkXStorage
from ._op import chunking_by_token_size, extract_entities, generate_community_report
from .base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage, StorageNameSpace


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
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 16
    embedding_func_max_async: int = 8

    # LLM
    best_model_func: callable = gpt_4o_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_func: callable = gpt_4o_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = MilvusLiteStorge
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    enable_llm_cache: bool = False

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

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
        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )

        self.text_chunks_vdb = None
        # self.text_chunks_vdb = self.vector_db_storage_cls(
        #     namespace="text_chunks",
        #     global_config=asdict(self),
        #     embedding_func=self.embedding_func,
        # )
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

    async def aquery(self, query: str):
        return await self.best_model_func(query)

    def query(self, query: str):
        return asyncio.run(self.aquery(query))

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        # TODO: no incremental update for communities now, so just drop all
        await self.community_reports.drop()
        # ---------- new docs
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            logger.warning(f"All docs are already in the storage")
            return
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")

        # ---------- chunking
        inserting_chunks = {}
        for doc_key, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
        _add_chunk_keys = await self.full_docs.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        # ---------- extract/summary entity and upsert to graph
        self.chunk_entity_relation_graph = await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            global_config=asdict(self),
        )
        logger.info("[Entity Extraction] Done")

        # ---------- update clusterings of graph
        await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
        await generate_community_report(
            self.community_reports, self.chunk_entity_relation_graph, asdict(self)
        )
        logger.info("[Community Report] Done")

        # ---------- commit upsertings and indexing
        await self.full_docs.upsert(new_docs)
        await self.text_chunks.upsert(chunks)

        await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.text_chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        pass

    def insert(self, string_or_strings):
        return asyncio.run(self.ainsert(string_or_strings))
