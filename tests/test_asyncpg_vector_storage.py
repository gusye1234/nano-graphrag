import numpy as np
import pytest
from dataclasses import asdict
from nano_graphrag import GraphRAG
from nano_graphrag._utils import wrap_embedding_func_with_attrs

from nano_graphrag.storage.asyncpg import AsyncpgVectorStorage
import asyncpg
from nano_graphrag.graphrag import always_get_an_event_loop
WORKING_DIR = "nano_graphrag_cache_asyncpg_vector_storage_test"
dsn='postgresql://test:test@127.0.0.1:12345/test'

@pytest.fixture(scope="function")
def setup_teardown():
    
    yield
    loop = always_get_an_event_loop()
    async def clean_table():
        conn: asyncpg.Connection = await asyncpg.connect(dsn)
        async with conn.transaction():
            tables = await conn.fetch(
                f"SELECT table_name FROM information_schema.tables WHERE table_name LIKE '{WORKING_DIR}%'"
            )

            for table in tables:
                await conn.execute(f"DROP TABLE {table['table_name']} CASCADE")
    loop.run_until_complete(clean_table())
    

@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


@pytest.fixture
def asyncpg_storage(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    return AsyncpgVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        dsn=dsn
    )


@pytest.mark.asyncio
async def test_upsert_and_query(asyncpg_storage):
    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
        "2": {"content": "Test content 2", "entity_name": "Entity 2"},
    }

    await asyncpg_storage.upsert(test_data)

    results = await asyncpg_storage.query("Test query", top_k=2)

    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all(
        "id" in result and "distance" in result and "similarity" in result
        for result in results
    )


@pytest.mark.asyncio
async def test_persistence(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    initial_storage = AsyncpgVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        dsn=dsn
    )

    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
    }

    await initial_storage.upsert(test_data)
    await initial_storage.index_done_callback()

    new_storage = AsyncpgVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        dsn=dsn
    )

    results = await new_storage.query("Test query", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert "entity_name" in results[0]


@pytest.mark.asyncio
async def test_persistence_large_dataset(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    initial_storage = AsyncpgVectorStorage(
        namespace="test_large",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        dsn=dsn
    )

    large_data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(1000)
    }
    await initial_storage.upsert(large_data)
    await initial_storage.index_done_callback()

    new_storage = AsyncpgVectorStorage(
        namespace="test_large",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        dsn=dsn
    )

    results = await new_storage.query("Test query", top_k=500)
    assert len(results) == 500
    assert all(result["id"] in large_data for result in results)


@pytest.mark.asyncio
async def test_upsert_with_existing_ids(asyncpg_storage):
    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
        "2": {"content": "Test content 2", "entity_name": "Entity 2"},
    }

    await asyncpg_storage.upsert(test_data)

    updated_data = {
        "1": {"content": "Updated content 1", "entity_name": "Updated Entity 1"},
        "3": {"content": "Test content 3", "entity_name": "Entity 3"},
    }

    await asyncpg_storage.upsert(updated_data)

    results = await asyncpg_storage.query("Updated", top_k=3)

    assert len(results) == 3
    assert any(
        result["id"] == "1" and result["entity_name"] == "Updated Entity 1"
        for result in results
    )
    assert any(
        result["id"] == "2" and result["entity_name"] == "Entity 2"
        for result in results
    )
    assert any(
        result["id"] == "3" and result["entity_name"] == "Entity 3"
        for result in results
    )


@pytest.mark.asyncio
async def test_large_batch_upsert(asyncpg_storage):
    batch_size = 30
    large_data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(batch_size)
    }

    await asyncpg_storage.upsert(large_data)

    results = await asyncpg_storage.query("Test query", top_k=batch_size)
    assert len(results) == batch_size
    assert all(isinstance(result, dict) for result in results)
    assert all(
        "id" in result and "distance" in result and "similarity" in result
        for result in results
    )


@pytest.mark.asyncio
async def test_empty_data_insertion(asyncpg_storage):
    empty_data = {}
    await asyncpg_storage.upsert(empty_data)

    results = await asyncpg_storage.query("Test query", top_k=1)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_query_with_no_results(asyncpg_storage):
    results = await asyncpg_storage.query("Non-existent query", top_k=5)
    assert len(results) == 0

    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
    }
    await asyncpg_storage.upsert(test_data)

    results = await asyncpg_storage.query("Non-existent query", top_k=5)
    assert len(results) == 1
    assert all(0 <= result["similarity"] <= 1 for result in results)
    assert "entity_name" in results[0]
