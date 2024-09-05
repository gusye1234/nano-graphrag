import os
import shutil
import numpy as np
import pytest
from unittest.mock import patch
from dataclasses import asdict
from nano_graphrag import GraphRAG
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from nano_graphrag._storage import HNSWVectorStorage

WORKING_DIR = "./tests/nano_graphrag_cache_hnsw_vector_storage_test"


@pytest.fixture(scope="function")
def setup_teardown():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

    yield

    shutil.rmtree(WORKING_DIR)


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


@pytest.fixture
def hnsw_storage(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    return HNSWVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
    )


@pytest.mark.asyncio
async def test_upsert_and_query(hnsw_storage):
    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
        "2": {"content": "Test content 2", "entity_name": "Entity 2"},
    }

    await hnsw_storage.upsert(test_data)

    results = await hnsw_storage.query("Test query", top_k=2)

    assert len(results) == 2
    assert all(isinstance(result, dict) for result in results)
    assert all(
        "id" in result and "distance" in result and "similarity" in result
        for result in results
    )


@pytest.mark.asyncio
async def test_persistence(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    initial_storage = HNSWVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
    )

    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
    }

    await initial_storage.upsert(test_data)
    await initial_storage.index_done_callback()

    new_storage = HNSWVectorStorage(
        namespace="test",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
    )

    results = await new_storage.query("Test query", top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert "entity_name" in results[0]


@pytest.mark.asyncio
async def test_persistence_large_dataset(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    initial_storage = HNSWVectorStorage(
        namespace="test_large",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        max_elements=10000,
    )

    large_data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(1000)
    }
    await initial_storage.upsert(large_data)
    await initial_storage.index_done_callback()

    new_storage = HNSWVectorStorage(
        namespace="test_large",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        max_elements=10000,
    )

    results = await new_storage.query("Test query", top_k=500)
    assert len(results) == 500
    assert all(result["id"] in large_data for result in results)


@pytest.mark.asyncio
async def test_upsert_with_existing_ids(hnsw_storage):
    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
        "2": {"content": "Test content 2", "entity_name": "Entity 2"},
    }

    await hnsw_storage.upsert(test_data)

    updated_data = {
        "1": {"content": "Updated content 1", "entity_name": "Updated Entity 1"},
        "3": {"content": "Test content 3", "entity_name": "Entity 3"},
    }

    await hnsw_storage.upsert(updated_data)

    results = await hnsw_storage.query("Updated", top_k=3)

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
async def test_large_batch_upsert(hnsw_storage):
    batch_size = 30
    large_data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(batch_size)
    }

    await hnsw_storage.upsert(large_data)

    results = await hnsw_storage.query("Test query", top_k=batch_size)
    assert len(results) == batch_size
    assert all(isinstance(result, dict) for result in results)
    assert all(
        "id" in result and "distance" in result and "similarity" in result
        for result in results
    )


@pytest.mark.asyncio
async def test_empty_data_insertion(hnsw_storage):
    empty_data = {}
    await hnsw_storage.upsert(empty_data)

    results = await hnsw_storage.query("Test query", top_k=1)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_query_with_no_results(hnsw_storage):
    results = await hnsw_storage.query("Non-existent query", top_k=5)
    assert len(results) == 0

    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
    }
    await hnsw_storage.upsert(test_data)

    results = await hnsw_storage.query("Non-existent query", top_k=5)
    assert len(results) == 1
    assert all(0 <= result["similarity"] <= 1 for result in results)
    assert "entity_name" in results[0]


@pytest.mark.asyncio
async def test_index_done_callback(hnsw_storage):
    test_data = {
        "1": {"content": "Test content 1", "entity_name": "Entity 1"},
    }

    await hnsw_storage.upsert(test_data)

    with patch("hnswlib.Index.save_index") as mock_save_index:
        await hnsw_storage.index_done_callback()
        mock_save_index.assert_called_once()


@pytest.mark.asyncio
async def test_max_elements_limit(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    max_elements = 10
    small_storage = HNSWVectorStorage(
        namespace="test_small",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        max_elements=max_elements,
        M=50,
    )

    data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(max_elements)
    }
    await small_storage.upsert(data)

    with pytest.raises(
        ValueError,
        match=f"Cannot insert 1 elements. Current: {max_elements}, Max: {max_elements}",
    ):
        await small_storage.upsert(
            {
                str(max_elements): {
                    "content": "Overflow",
                    "entity_name": "Overflow Entity",
                }
            }
        )

    large_max_elements = 100
    large_storage = HNSWVectorStorage(
        namespace="test_large",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        max_elements=large_max_elements,
    )

    initial_data_size = int(large_max_elements * 0.3)
    initial_data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(initial_data_size)
    }

    await large_storage.upsert(initial_data)

    results = await large_storage.query("Test query", top_k=initial_data_size)
    assert len(results) == initial_data_size


@pytest.mark.asyncio
async def test_ef_search_values(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    storage = HNSWVectorStorage(
        namespace="test_ef",
        global_config=asdict(rag),
        embedding_func=mock_embedding,
        meta_fields={"entity_name"},
        ef_search=10,
    )

    data = {
        str(i): {"content": f"Test content {i}", "entity_name": f"Entity {i}"}
        for i in range(20)
    }
    await storage.upsert(data)

    results_default = await storage.query("Test query", top_k=5)
    assert len(results_default) == 5

    storage._index.set_ef(20)
    results_higher_ef = await storage.query("Test query", top_k=15)
    assert len(results_higher_ef) == 15
