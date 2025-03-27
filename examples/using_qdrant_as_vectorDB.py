import os
import asyncio
import uuid
import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import logger
from nano_graphrag.base import BaseVectorStorage
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, SearchParams
except ImportError as original_error:
    raise ImportError(
        "Qdrant client is not installed. Install it using: pip install qdrant-client\n"
    ) from original_error


@dataclass
class QdrantStorage(BaseVectorStorage):
    def __post_init__(self):

        # Use a local file-based Qdrant storage
        # Useful for prototyping and CI.
        # For production, refer to:
        # https://qdrant.tech/documentation/guides/installation/
        self._client_file_path = os.path.join(
            self.global_config["working_dir"], "qdrant_storage"
        )

        self._client = QdrantClient(path=self._client_file_path)

        self._max_batch_size = self.global_config["embedding_batch_num"]

        if not self._client.collection_exists(collection_name=self.namespace):
            self._client.create_collection(
                collection_name=self.namespace,
                vectors_config=VectorParams(
                    size=self.embedding_func.embedding_dim, distance=Distance.COSINE
                ),
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

        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=embeddings[i].tolist(),
                payload=data,
            )
            for i, data in enumerate(list_data)
        ]

        results = self._client.upsert(collection_name=self.namespace, points=points)
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])

        results = self._client.query_points(
            collection_name=self.namespace,
            query=embedding[0].tolist(),
            limit=top_k,
        ).points

        return [
            {**result.payload, "score": result.score}
            for result in results
        ]


def insert():
    data = ["YOUR TEXT DATA HERE", "YOUR TEXT DATA HERE"]
    rag = GraphRAG(
        working_dir="./nano_graphrag_cache_qdrant_TEST",
        enable_llm_cache=True,
        vector_db_storage_cls=QdrantStorage,
    )
    rag.insert(data)


def query():
    rag = GraphRAG(
        working_dir="./nano_graphrag_cache_qdrant_TEST",
        enable_llm_cache=True,
        vector_db_storage_cls=QdrantStorage,
    )
    print(rag.query("YOUR QUERY HERE", param=QueryParam(mode="local")))


if __name__ == "__main__":
    insert()
    query()
