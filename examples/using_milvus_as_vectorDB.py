import os
import asyncio
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import logger
from nano_graphrag.base import BaseVectorStorage
from dataclasses import dataclass


@dataclass
class MilvusLiteStorge(BaseVectorStorage):

    @staticmethod
    def create_collection_if_not_exist(client, collection_name: str, **kwargs):
        if client.has_collection(collection_name):
            return
        # TODO add constants for ID max length to 32
        client.create_collection(
            collection_name, max_length=32, id_type="string", **kwargs
        )

    def __post_init__(self):
        from pymilvus import MilvusClient

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], "milvus_lite.db"
        )
        self._client = MilvusClient(self._client_file_name)
        self._max_batch_size = self.global_config["embedding_batch_num"]
        MilvusLiteStorge.create_collection_if_not_exist(
            self._client,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
        )
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]


def insert():
    data = ["YOUR TEXT DATA HERE", "YOUR TEXT DATA HERE"]
    rag = GraphRAG(
        working_dir="./nano_graphrag_cache_milvus_TEST",
        enable_llm_cache=True,
        vector_db_storage_cls=MilvusLiteStorge,
    )
    rag.insert(data)


def query():
    rag = GraphRAG(
        working_dir="./nano_graphrag_cache_milvus_TEST",
        enable_llm_cache=True,
        vector_db_storage_cls=MilvusLiteStorge,
    )
    print(rag.query("YOUR QUERY HERE", param=QueryParam(mode="local")))


insert()
query()
