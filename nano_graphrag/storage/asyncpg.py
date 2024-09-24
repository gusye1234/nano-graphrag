from nano_graphrag._storage import BaseVectorStorage
import asyncpg
import asyncio
from contextlib import asynccontextmanager
from nano_graphrag._utils import logger
from pgvector.asyncpg import register_vector
from nano_graphrag.graphrag import always_get_an_event_loop
import numpy as np
import json
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

@dataclass
class AsyncPGVectorStorage(BaseVectorStorage):
    table_name_generator: callable = None
    conn_fetcher: callable = None
    cosine_better_than_threshold: float = 0.2
    dsn = None
    def __post_init__(self):
        params = self.global_config.get("vector_db_storage_cls_kwargs", {})
        dsn = params.get("dsn", None)
        conn_fetcher = params.get("conn_fetcher", None)
        table_name_generator = params.get("table_name_generator", None)
        self.dsn = dsn
        self.conn_fetcher = conn_fetcher
        assert self.dsn != None or self.conn_fetcher != None, "Must provide either dsn or conn_fetcher"
        if self.dsn:
            self.conn_fetcher = self.__get_conn
        if not table_name_generator:
            self.table_name_generator = lambda working_dir, namespace: f'{working_dir}_{namespace}_vdb'
        self._table_name = self.table_name_generator(self.global_config["working_dir"], self.namespace)
        self._max_batch_size = self.global_config["embedding_batch_num"]
        
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )
        loop = always_get_an_event_loop()
        loop.run_until_complete(self._secure_table())
    @asynccontextmanager
    async def __get_conn(self, vector_register=True):
        try:
            conn: asyncpg.Connection = await asyncpg.connect(self.dsn)
            if vector_register:
                await register_vector(conn)
            yield conn
        finally:
            await conn.close()
    async def _secure_table(self):
        async with self.conn_fetcher(vector_register=False) as conn:
            conn: asyncpg.Connection
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            result = await conn.fetch(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)", self._table_name)
            table_exists = result[0]['exists']
            if not table_exists:
                # create the table
                await conn.execute(f'CREATE TABLE {self._table_name} (id text PRIMARY KEY, embedding vector({self.embedding_func.embedding_dim}), data jsonb)')
                await conn.execute(f'CREATE INDEX ON {self._table_name} USING hnsw (embedding vector_cosine_ops)')
    async def query(self, query: str, top_k: int) -> list[dict]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        async with self.conn_fetcher() as conn:

            result = await conn.fetch(f'SELECT embedding <=> $1 as similarity, id, embedding, data FROM {self._table_name} WHERE embedding <=> $1 > $3 ORDER BY embedding <=> $1 DESC LIMIT $2', embedding, top_k, self.cosine_better_than_threshold)

            rows = []
            for row in result:
                data = json.loads(row['data'])
                rows.append({
                    **data,
                    'id': row['id'],
                    'distance': 1 - row['similarity'],
                    'similarity': row['similarity']
                })
            return rows
    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
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
        embeddings_list = np.concatenate(embeddings_list)
        insert_rows = []
        for i, d in enumerate(list_data):
            row = [d["__id__"], embeddings_list[i], json.dumps(d)]
            insert_rows.append(row)
        async with self.conn_fetcher() as conn:
            conn: asyncpg.Connection
            stmt = f"INSERT INTO {self._table_name} (id, embedding, data) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET embedding = $2, data = $3"
            return await conn.executemany(stmt, insert_rows)
