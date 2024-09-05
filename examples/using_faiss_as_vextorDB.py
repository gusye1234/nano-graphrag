import os
import asyncio
import numpy as np
from nano_graphrag.graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import logger
from nano_graphrag.base import BaseVectorStorage
from dataclasses import dataclass
import faiss
import pickle
import logging
import xxhash
logging.getLogger('msal').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

WORKING_DIR = "./nano_graphrag_cache_faiss_TEST"

@dataclass
class FAISSStorage(BaseVectorStorage):

    def __post_init__(self):
        self._index_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_faiss.index"
        )
        self._metadata_file_name = os.path.join(
            self.global_config["working_dir"], f"{self.namespace}_metadata.pkl"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        if os.path.exists(self._index_file_name) and os.path.exists(self._metadata_file_name):
            self._index = faiss.read_index(self._index_file_name)
            with open(self._metadata_file_name, 'rb') as f:
                self._metadata = pickle.load(f)
        else:
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_func.embedding_dim))
            self._metadata = {}

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        
        ids = []
        for k, v in data.items():
            id = xxhash.xxh32_intdigest(k.encode())
            metadata = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            metadata['id'] = k
            self._metadata[id] = metadata
            ids.append(id)
        
        ids = np.array(ids, dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)
        
        
        return len(data)

    async def query(self, query, top_k=5):
        embedding = await self.embedding_func([query])
        distances, indices = self._index.search(embedding, top_k)
        
        results = []
        for _, (distance, id) in enumerate(zip(distances[0], indices[0])):
            if id != -1:  # FAISS returns -1 for empty slots
                if id in self._metadata:
                    metadata = self._metadata[id]
                    results.append({**metadata, "distance": 1 - distance})  # Convert to cosine distance
        
        return results
    
    async def index_done_callback(self):
        faiss.write_index(self._index, self._index_file_name)
        with open(self._metadata_file_name, 'wb') as f:
            pickle.dump(self._metadata, f)

if __name__ == "__main__":

    graph_func = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        vector_db_storage_cls=FAISSStorage,
    )

    with open(r"tests/mock_data.txt", encoding='utf-8') as f:
        graph_func.insert(f.read()[:30000])

    # Perform global graphrag search
    print(graph_func.query("What are the top themes in this story?"))

    