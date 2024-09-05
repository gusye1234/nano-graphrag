import os
import shutil
import pytest
import networkx as nx
import numpy as np
import asyncio
from nano_graphrag import GraphRAG
from nano_graphrag._storage import NetworkXStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

WORKING_DIR = "./tests/nano_graphrag_cache_networkx_storage_test"


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
def networkx_storage(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    return NetworkXStorage(
        namespace="test",
        global_config=rag.__dict__,
    )


@pytest.mark.asyncio
async def test_upsert_and_get_node(networkx_storage):
    node_id = "node1"
    node_data = {"attr1": "value1", "attr2": "value2"}
    
    await networkx_storage.upsert_node(node_id, node_data)
    
    result = await networkx_storage.get_node(node_id)
    assert result == node_data
    
    has_node = await networkx_storage.has_node(node_id)
    assert has_node is True


@pytest.mark.asyncio
async def test_upsert_and_get_edge(networkx_storage):
    source_id = "node1"
    target_id = "node2"
    edge_data = {"weight": 1.0, "type": "connection"}
    
    await networkx_storage.upsert_node(source_id, {})
    await networkx_storage.upsert_node(target_id, {})
    await networkx_storage.upsert_edge(source_id, target_id, edge_data)
    
    result = await networkx_storage.get_edge(source_id, target_id)
    assert result == edge_data
    
    has_edge = await networkx_storage.has_edge(source_id, target_id)
    assert has_edge is True


@pytest.mark.asyncio
async def test_node_degree(networkx_storage):
    node_id = "center"
    await networkx_storage.upsert_node(node_id, {})
    
    num_neighbors = 5
    for i in range(num_neighbors):
        neighbor_id = f"neighbor{i}"
        await networkx_storage.upsert_node(neighbor_id, {})
        await networkx_storage.upsert_edge(node_id, neighbor_id, {})
    
    degree = await networkx_storage.node_degree(node_id)
    assert degree == num_neighbors


@pytest.mark.asyncio
async def test_edge_degree(networkx_storage):
    source_id = "node1"
    target_id = "node2"
    
    await networkx_storage.upsert_node(source_id, {})
    await networkx_storage.upsert_node(target_id, {})
    await networkx_storage.upsert_edge(source_id, target_id, {})
    
    num_source_neighbors = 3
    for i in range(num_source_neighbors):
        neighbor_id = f"neighbor{i}"
        await networkx_storage.upsert_node(neighbor_id, {})
        await networkx_storage.upsert_edge(source_id, neighbor_id, {})
    
    num_target_neighbors = 2
    for i in range(num_target_neighbors):
        neighbor_id = f"target_neighbor{i}"
        await networkx_storage.upsert_node(neighbor_id, {})
        await networkx_storage.upsert_edge(target_id, neighbor_id, {})
    
    expected_edge_degree = (num_source_neighbors + 1) + (num_target_neighbors + 1)
    edge_degree = await networkx_storage.edge_degree(source_id, target_id)
    assert edge_degree == expected_edge_degree


@pytest.mark.asyncio
async def test_get_node_edges(networkx_storage):
    center_id = "center"
    await networkx_storage.upsert_node(center_id, {})
    
    expected_edges = []
    for i in range(3):
        neighbor_id = f"neighbor{i}"
        await networkx_storage.upsert_node(neighbor_id, {})
        await networkx_storage.upsert_edge(center_id, neighbor_id, {})
        expected_edges.append((center_id, neighbor_id))
    
    result = await networkx_storage.get_node_edges(center_id)
    assert set(result) == set(expected_edges)


@pytest.mark.parametrize("algorithm", ["leiden"])
@pytest.mark.asyncio
async def test_clustering(networkx_storage, algorithm):
    # [numberchiffre]: node ID is case-sensitive for clustering with leiden.
    for i in range(10):
        await networkx_storage.upsert_node(f"NODE{i}", {"source_id": f"chunk{i}"})
    
    for i in range(9):
        await networkx_storage.upsert_edge(f"NODE{i}", f"NODE{i+1}", {})
    
    assert networkx_storage._graph.number_of_nodes() > 0
    assert networkx_storage._graph.number_of_edges() > 0
    await networkx_storage.clustering(algorithm=algorithm)
    
    community_schema = await networkx_storage.community_schema()

    assert len(community_schema) > 0
    
    for community in community_schema.values():
        assert "level" in community
        assert "title" in community
        assert "edges" in community
        assert "nodes" in community
        assert "chunk_ids" in community
        assert "occurrence" in community
        assert "sub_communities" in community


@pytest.mark.asyncio
async def test_persistence(setup_teardown):
    rag = GraphRAG(working_dir=WORKING_DIR, embedding_func=mock_embedding)
    initial_storage = NetworkXStorage(
        namespace="test_persistence",
        global_config=rag.__dict__,
    )
    
    await initial_storage.upsert_node("node1", {"attr": "value"})
    await initial_storage.upsert_node("node2", {"attr": "value"})
    await initial_storage.upsert_edge("node1", "node2", {"weight": 1.0})
    
    await initial_storage.index_done_callback()
    
    new_storage = NetworkXStorage(
        namespace="test_persistence",
        global_config=rag.__dict__,
    )
    
    assert await new_storage.has_node("node1")
    assert await new_storage.has_node("node2")
    assert await new_storage.has_edge("node1", "node2")
    
    node1_data = await new_storage.get_node("node1")
    assert node1_data == {"attr": "value"}
    
    edge_data = await new_storage.get_edge("node1", "node2")
    assert edge_data == {"weight": 1.0}


@pytest.mark.asyncio
async def test_embed_nodes(networkx_storage):
    for i in range(5):
        await networkx_storage.upsert_node(f"node{i}", {"id": f"node{i}"})
    
    for i in range(4):
        await networkx_storage.upsert_edge(f"node{i}", f"node{i+1}", {})
    
    embeddings, node_ids = await networkx_storage.embed_nodes("node2vec")
    
    assert embeddings.shape == (5, networkx_storage.global_config['node2vec_params']['dimensions'])
    assert len(node_ids) == 5
    assert all(f"node{i}" in node_ids for i in range(5))


@pytest.mark.asyncio
async def test_stable_largest_connected_component_equal_components():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("C", "D"), ("E", "F")])
    result = NetworkXStorage.stable_largest_connected_component(G)
    assert sorted(result.nodes()) == ["A", "B"]
    assert list(result.edges()) == [("A", "B")]


@pytest.mark.asyncio
async def test_stable_largest_connected_component_directed_graph():
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])
    result = NetworkXStorage.stable_largest_connected_component(G)
    assert sorted(result.nodes()) == ["A", "B", "C", "D"]
    assert sorted(result.edges()) == [("A", "B"), ("B", "C"), ("C", "D")]


@pytest.mark.asyncio
async def test_stable_largest_connected_component_self_loops_and_parallel_edges():
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "A"), ("B", "B"), ("A", "B")])
    result = NetworkXStorage.stable_largest_connected_component(G)
    assert sorted(result.nodes()) == ["A", "B", "C"]
    assert sorted(result.edges()) == [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C')]


@pytest.mark.asyncio
async def test_community_schema_with_no_clusters(networkx_storage):
    await networkx_storage.upsert_node("node1", {"source_id": "chunk1"})
    await networkx_storage.upsert_node("node2", {"source_id": "chunk2"})
    await networkx_storage.upsert_edge("node1", "node2", {})
    
    community_schema = await networkx_storage.community_schema()
    assert len(community_schema) == 0


@pytest.mark.asyncio
async def test_concurrent_operations(networkx_storage):
    async def add_nodes(start, end):
        for i in range(start, end):
            await networkx_storage.upsert_node(f"node{i}", {"value": i})

    await asyncio.gather(
        add_nodes(0, 500),
        add_nodes(500, 1000)
    )

    assert await networkx_storage.node_degree("node0") == 0
    assert len(networkx_storage._graph.nodes) == 1000


@pytest.mark.asyncio
async def test_nonexistent_node_and_edge(networkx_storage):
    assert await networkx_storage.has_node("nonexistent") is False
    assert await networkx_storage.has_edge("node1", "node2") is False
    assert await networkx_storage.get_node("nonexistent") is None
    assert await networkx_storage.get_edge("node1", "node2") is None
    assert await networkx_storage.get_node_edges("nonexistent") is None
    assert await networkx_storage.node_degree("nonexistent") == 0
    assert await networkx_storage.edge_degree("node1", "node2") == 0


@pytest.mark.asyncio
async def test_error_handling(networkx_storage):
    with pytest.raises(ValueError, match="Clustering algorithm invalid_algo not supported"):
        await networkx_storage.clustering("invalid_algo")

    with pytest.raises(ValueError, match="Node embedding algorithm invalid_algo not supported"):
        await networkx_storage.embed_nodes("invalid_algo")
