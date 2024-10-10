import asyncio
import os
from dataclasses import dataclass
import numpy as np
from pymilvus import MilvusClient

from .._utils import get_workdir_last_folder_name, logger
from ..base import BaseVectorStorage


@dataclass
class MilvusVectorStorage(BaseVectorStorage):

    @staticmethod
    def create_collection_if_not_exist(client, collection_name: str,max_id_length: int, dimension: int,**kwargs):
        if client.has_collection(collection_name):
            return
        client.create_collection(
            collection_name, max_length=max_id_length, id_type="string", dimension=dimension, **kwargs
        )


    def __post_init__(self):
        self.milvus_uri = self.global_config["addon_params"].get("milvus_uri", "")
        if self.milvus_uri:
            self.milvus_user = self.global_config["addon_params"].get("milvus_user", "")
            self.milvus_password = self.global_config["addon_params"].get("milvus_password", "")
            self.collection_name = get_workdir_last_folder_name(self.global_config["working_dir"])
            self._client = MilvusClient(self.milvus_uri, self.milvus_user, self.milvus_password)
        else:
            self._client_file_name = os.path.join(
                self.global_config["working_dir"], "milvus_lite.db"
            )
            self._client = MilvusClient(self._client_file_name)

        self.cosine_better_than_threshold: float = 0.2
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.max_id_length = 256
        MilvusVectorStorage.create_collection_if_not_exist(
            self._client, self.collection_name,max_id_length=self.max_id_length,dimension=self.embedding_func.embedding_dim, 
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.collection_name}")
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
        batch_size = 1024
        results = []
        for i in range(0, len(list_data), batch_size):
            batch = list_data[i:i+batch_size]
            batch_result = self._client.upsert(collection_name=self.collection_name, data=batch)
            results.append(batch_result)
        
        total_upsert_count = sum(result.get('upsert_count', 0) for result in results)
        results = {'upsert_count': total_upsert_count}
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.collection_name,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": self.cosine_better_than_threshold}},
        )
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]
