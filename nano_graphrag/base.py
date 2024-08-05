from typing import Union
from dataclasses import dataclass, field
from ._utils import EmbeddingFunc


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query):
        raise NotImplementedError

    async def insert(self, data: dict[str, dict]):
        """Use 'content' from value for embedding, use key as id"""
        raise NotImplementedError


@dataclass
class BaseKVStorage(StorageNameSpace):
    async def get_by_id(self, id) -> Union[dict, None]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError
