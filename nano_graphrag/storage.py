import os
import asyncio
import numpy as np
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from ._utils import load_json, write_json, EmbeddingFunc


@dataclass
class BaseVectorStorage:
    namespace: str
    global_config: dict
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query):
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' from value for embedding, use key as id"""
        raise NotImplementedError


@dataclass
class BaseKVStorage:
    namespace: str
    global_config: dict

    async def get_by_id(self, id):
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}

    async def get_by_id(self, id):
        return self._data[id]

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)
        write_json(self._data, self._file_name)


@dataclass
class MilvusLiteStorge(BaseVectorStorage):
    @staticmethod
    def create_collection_if_not_exist(
        client: MilvusClient, collection_name: str, **kwargs
    ):
        if client.has_collection(collection_name):
            return
        # TODO add constants for ID max length to 32
        client.create_collection(
            collection_name, max_length=32, id_type="string", auto_id=False, **kwargs
        )

    def __post_init__(self):
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
        results = self._client.insert(collection_name=self.namespace, data=list_data)
        return results

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        results = self._client.search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
        )
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]
