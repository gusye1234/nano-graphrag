import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
import pickle
import hnswlib
import numpy as np
import xxhash

from .._utils import logger
from ..base import BaseVectorStorage


@dataclass
class HNSWVectorStorage(BaseVectorStorage):
    ef_construction: int = 100
    M: int = 16
    max_elements: int = 1000000
    ef_search: int = 50
    num_threads: int = -1
    _index: Any = field(init=False)
    _metadata: dict[str, dict] = field(default_factory=dict)
    _current_elements: int = 0

    def __post_init__(self):
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_hnsw_metadata.pkl"
        )
        self._embedding_batch_num = self.global_config.get("embedding_batch_num", 100)

        hnsw_params = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.ef_construction = hnsw_params.get("ef_construction", self.ef_construction)
        self.M = hnsw_params.get("M", self.M)
        self.max_elements = hnsw_params.get("max_elements", self.max_elements)
        self.ef_search = hnsw_params.get("ef_search", self.ef_search)
        self.num_threads = hnsw_params.get("num_threads", self.num_threads)
        self._index = hnswlib.Index(
            space="cosine", dim=self.embedding_func.embedding_dim
        )

        if os.path.exists(self._index_file_name) and os.path.exists(
            self._metadata_file_name
        ):
            self._index.load_index(
                self._index_file_name, max_elements=self.max_elements
            )
            with open(self._metadata_file_name, "rb") as f:
                self._metadata, self._current_elements = pickle.load(f)
            logger.info(
                f"Loaded existing index for {self.namespace} with {self._current_elements} elements"
            )
        else:
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(self.ef_search)
            self._metadata = {}
            self._current_elements = 0
            logger.info(f"Created new index for {self.namespace}")

    async def upsert(self, data: dict[str, dict]) -> np.ndarray:
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You insert an empty data to vector DB")
            return []

        if self._current_elements + len(data) > self.max_elements:
            raise ValueError(
                f"Cannot insert {len(data)} elements. Current: {self._current_elements}, Max: {self.max_elements}"
            )

        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batch_size = min(self._embedding_batch_num, len(contents))
        embeddings = np.concatenate(
            await asyncio.gather(
                *[
                    self.embedding_func(contents[i : i + batch_size])
                    for i in range(0, len(contents), batch_size)
                ]
            )
        )

        ids = np.fromiter(
            (xxhash.xxh32_intdigest(d["id"].encode()) for d in list_data),
            dtype=np.uint32,
            count=len(list_data),
        )
        self._metadata.update(
            {
                id_int: {
                    k: v for k, v in d.items() if k in self.meta_fields or k == "id"
                }
                for id_int, d in zip(ids, list_data)
            }
        )
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.num_threads)
        self._current_elements = self._index.get_current_count()
        return ids

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        if self._current_elements == 0:
            return []

        top_k = min(top_k, self._current_elements)

        if top_k > self.ef_search:
            logger.warning(
                f"Setting ef_search to {top_k} because top_k is larger than ef_search"
            )
            self._index.set_ef(top_k)

        embedding = await self.embedding_func([query])
        labels, distances = self._index.knn_query(
            data=embedding[0], k=top_k, num_threads=self.num_threads
        )

        return [
            {
                **self._metadata.get(label, {}),
                "distance": distance,
                "similarity": 1 - distance,
            }
            for label, distance in zip(labels[0], distances[0])
        ]

    async def index_done_callback(self):
        self._index.save_index(self._index_file_name)
        with open(self._metadata_file_name, "wb") as f:
            pickle.dump((self._metadata, self._current_elements), f)
