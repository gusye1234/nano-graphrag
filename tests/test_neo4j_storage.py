import os
import pytest
import numpy as np
from functools import wraps
from nano_graphrag import GraphRAG
from nano_graphrag._storage import Neo4jStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

if os.environ.get("NANO_GRAPHRAG_TEST_IGNORE_NEO4J", False):
    pytest.skip("skipping neo4j tests", allow_module_level=True)


@pytest.fixture(scope="module")
def neo4j_config():
    return {
        "neo4j_url": os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
        "neo4j_auth": (
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4j"),
        ),
    }


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


@pytest.fixture
def neo4j_storage(neo4j_config):
    rag = GraphRAG(
        working_dir="./tests/neo4j_test",
        embedding_func=mock_embedding,
        graph_storage_cls=Neo4jStorage,
        addon_params=neo4j_config,
    )
    storage = rag.chunk_entity_relation_graph
    return storage


def reset_graph(func):
    @wraps(func)
    async def new_func(neo4j_storage):
        await neo4j_storage._debug_delete_all_node_edges()
        await neo4j_storage.index_start_callback()
        results = await func(neo4j_storage)
        await neo4j_storage._debug_delete_all_node_edges()
        return results

    return new_func


def test_neo4j_storage_init():
    rag = GraphRAG(
        working_dir="./tests/neo4j_test",
        embedding_func=mock_embedding,
    )
    with pytest.raises(ValueError):
        storage = Neo4jStorage(
            namespace="nanographrag_test", global_config=rag.__dict__
        )


@pytest.mark.asyncio
@reset_graph
async def test_upsert_and_get_node(neo4j_storage):
    node_id = "node1"
    node_data = {"attr1": "value1", "attr2": "value2"}
    return_data = {"id": node_id, "clusters": "[]", **node_data}

    await neo4j_storage.upsert_node(node_id, node_data)

    result = await neo4j_storage.get_node(node_id)
    assert result == return_data

    has_node = await neo4j_storage.has_node(node_id)
    assert has_node is True


@pytest.mark.asyncio
@reset_graph
async def test_upsert_and_get_edge(neo4j_storage):
    source_id = "node1"
    target_id = "node2"
    edge_data = {"weight": 1.0, "type": "connection"}

    await neo4j_storage.upsert_node(source_id, {})
    await neo4j_storage.upsert_node(target_id, {})
    await neo4j_storage.upsert_edge(source_id, target_id, edge_data)

    result = await neo4j_storage.get_edge(source_id, target_id)
    print(result)
    assert result == edge_data

    has_edge = await neo4j_storage.has_edge(source_id, target_id)
    assert has_edge is True


@pytest.mark.asyncio
@reset_graph
async def test_node_degree(neo4j_storage):
    node_id = "center"
    await neo4j_storage.upsert_node(node_id, {})

    num_neighbors = 5
    for i in range(num_neighbors):
        neighbor_id = f"neighbor{i}"
        await neo4j_storage.upsert_node(neighbor_id, {})
        await neo4j_storage.upsert_edge(node_id, neighbor_id, {})

    degree = await neo4j_storage.node_degree(node_id)
    assert degree == num_neighbors


@pytest.mark.asyncio
@reset_graph
async def test_edge_degree(neo4j_storage):
    source_id = "node1"
    target_id = "node2"

    await neo4j_storage.upsert_node(source_id, {})
    await neo4j_storage.upsert_node(target_id, {})
    await neo4j_storage.upsert_edge(source_id, target_id, {})

    num_source_neighbors = 3
    for i in range(num_source_neighbors):
        neighbor_id = f"neighbor{i}"
        await neo4j_storage.upsert_node(neighbor_id, {})
        await neo4j_storage.upsert_edge(source_id, neighbor_id, {})

    num_target_neighbors = 2
    for i in range(num_target_neighbors):
        neighbor_id = f"target_neighbor{i}"
        await neo4j_storage.upsert_node(neighbor_id, {})
        await neo4j_storage.upsert_edge(target_id, neighbor_id, {})

    expected_edge_degree = (num_source_neighbors + 1) + (num_target_neighbors + 1)
    edge_degree = await neo4j_storage.edge_degree(source_id, target_id)
    assert edge_degree == expected_edge_degree


@pytest.mark.asyncio
@reset_graph
async def test_get_node_edges(neo4j_storage):
    center_id = "center"
    await neo4j_storage.upsert_node(center_id, {})

    expected_edges = []
    for i in range(3):
        neighbor_id = f"neighbor{i}"
        await neo4j_storage.upsert_node(neighbor_id, {})
        await neo4j_storage.upsert_edge(center_id, neighbor_id, {})
        expected_edges.append((center_id, neighbor_id))

    result = await neo4j_storage.get_node_edges(center_id)
    print(result)
    assert set(result) == set(expected_edges)


@pytest.mark.asyncio
@reset_graph
async def test_leiden_clustering(neo4j_storage):
    for i in range(10):
        await neo4j_storage.upsert_node(f"NODE{i}", {"source_id": f"chunk{i}"})

    for i in range(9):
        await neo4j_storage.upsert_edge(f"NODE{i}", f"NODE{i+1}", {"weight": 1.0})

    await neo4j_storage.clustering(algorithm="leiden")

    community_schema = await neo4j_storage.community_schema()

    assert len(community_schema) > 0

    for community in community_schema.values():
        assert "level" in community
        assert "title" in community
        assert "edges" in community
        assert "nodes" in community
        assert "chunk_ids" in community
        assert "occurrence" in community
        assert "sub_communities" in community
        print(community)


@pytest.mark.asyncio
@reset_graph
async def test_nonexistent_node_and_edge(neo4j_storage):
    assert await neo4j_storage.has_node("nonexistent") is False
    assert await neo4j_storage.has_edge("node1", "node2") is False
    assert await neo4j_storage.get_node("nonexistent") is None
    assert await neo4j_storage.get_edge("node1", "node2") is None
    assert await neo4j_storage.get_node_edges("nonexistent") == []
    assert await neo4j_storage.node_degree("nonexistent") == 0
    assert await neo4j_storage.edge_degree("node1", "node2") == 0


@pytest.mark.asyncio
@reset_graph
async def test_cluster_error_handling(neo4j_storage):
    with pytest.raises(
        ValueError, match="Clustering algorithm invalid_algo not supported"
    ):
        await neo4j_storage.clustering("invalid_algo")


@pytest.mark.asyncio
@reset_graph
async def test_index_done(neo4j_storage):
    await neo4j_storage.index_done_callback()
