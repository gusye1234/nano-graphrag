from functools import wraps
import os
import pytest
import numpy as np
import json
from nano_graphrag import GraphRAG
from nano_graphrag._storage import NebulaGraphStorage
from nano_graphrag._utils import wrap_embedding_func_with_attrs

if os.environ.get("NANO_GRAPHRAG_TEST_IGNORE_NEBULA", False):
    pytest.skip("skipping nebula tests", allow_module_level=True)

@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=8192)
async def mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 384)


@pytest.fixture(scope="module")
def nebula_config():
    return {
        "graphd_hosts": os.environ.get("NEBULA_GRAPHD_HOSTS", "localhost:9669"),
        "metad_hosts": os.environ.get("NEBULA_METAD_HOSTS","localhost:9559"),
        "username": os.environ.get("NEBULA_USERNAME", "root"),
        "password": os.environ.get("NEBULA_PASSWORD", "nebula"),
    }


@pytest.fixture
def nebula_graph_storage(nebula_config):
    rag = GraphRAG(
        working_dir="./tests/nebula_test",
        embedding_func=mock_embedding,
        graph_storage_cls=NebulaGraphStorage,
        community_report_storage_cls = NebulaGraphStorage,
        addon_params=nebula_config,
    )
    storage = rag.chunk_entity_relation_graph
    return storage

@pytest.fixture
def nebula_kv_storage(nebula_config):
    rag = GraphRAG(
        working_dir="./tests/nebula_test",
        embedding_func=mock_embedding,
        graph_storage_cls=NebulaGraphStorage,
        community_report_storage_cls = NebulaGraphStorage,
        addon_params=nebula_config,
    )
    storage = rag.community_reports
    return storage


def test_nebula_storage_init():
    rag = GraphRAG(
        working_dir="./tests/neo4j_test",
        embedding_func=mock_embedding,
    )
    with pytest.raises(ValueError):
        storage = NebulaGraphStorage(
            namespace="nanographrag_test", global_config=rag.__dict__
        )

def delete_all_data(nebula_graph_storage):
    """Delete all tags and edges from the Nebula Graph database."""
    TAG_NAMES = [nebula_graph_storage.INIT_VERTEX_TYPE, nebula_graph_storage.COMMUNITY_VERTEX_TYPE]

    try:
        for tag in TAG_NAMES:
            delete_query = f"LOOKUP ON {tag} YIELD id(vertex) AS ID | DELETE VERTEX  $-.ID WITH EDGE;"
            result = nebula_graph_storage.client.execute_py(delete_query)
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to delete vertices: {result.error_msg()}")
        
    except Exception as e:
        print(f"Error occurred while deleting data: {e}")

def delete_space(nebula_graph_storage):
    """Delete the space from the Nebula Graph database."""
    try:
        # Get the current space name
        space_name = nebula_graph_storage.space

        # Drop the specified space
        drop_space_query = f"DROP SPACE IF EXISTS {space_name}"
        result = nebula_graph_storage.client.execute_py(drop_space_query)
        if not result.is_succeeded():
            raise RuntimeError(f"Failed to drop space: {result.error_msg()}")
        
        print(f"Successfully dropped space: {space_name}")
    except Exception as e:
        print(f"Error occurred while dropping space: {e}")


def reset_graph(func):
    @wraps(func)
    async def new_func(nebula_graph_storage, *args, **kwargs):
        delete_all_data(nebula_graph_storage)
        results = await func(nebula_graph_storage, *args, **kwargs)
        delete_all_data(nebula_graph_storage)
        return results

    return new_func


@pytest.mark.asyncio
@reset_graph
async def test_upsert_and_get_node(nebula_graph_storage):
    node_id = "node1"
    node_data = {"entity_name": "Entity 1", "description": "description111"}
    return_data = {"id": node_id, **node_data}

    await nebula_graph_storage.upsert_node(node_id, node_data)

    result = await nebula_graph_storage.get_node(node_id)
    assert all(result.get(key) == value for key, value in return_data.items())

    has_node = await nebula_graph_storage.has_node(node_id)
    assert has_node is True

    non_existent_node = await nebula_graph_storage.get_node("non_existent")
    assert non_existent_node is None

    has_non_existent_node = await nebula_graph_storage.has_node("non_existent")
    assert has_non_existent_node is False
    
 


@pytest.mark.asyncio
@reset_graph
async def test_upsert_and_get_edge(nebula_graph_storage):
    source_id = "node1"
    target_id = "node2"
    node_data = {"entity_name": "Entity 1", "description": "description111"}
    edge_data = {"weight": 1.0, "description": "connection"}

    await nebula_graph_storage.upsert_node(source_id, node_data)
    await nebula_graph_storage.upsert_node(target_id, node_data)
    await nebula_graph_storage.upsert_edge(source_id, target_id, edge_data)

    result = await nebula_graph_storage.get_edge(source_id, target_id)
    assert all(result.get(key) == value for key, value in edge_data.items())

    has_edge = await nebula_graph_storage.has_edge(source_id, target_id)
    assert has_edge is True

    non_existent_edge = await nebula_graph_storage.get_edge("non_existent1", "non_existent2")
    assert non_existent_edge is None

    has_non_existent_edge = await nebula_graph_storage.has_edge("non_existent1", "non_existent2")
    assert has_non_existent_edge is False
    
 


@pytest.mark.asyncio
@reset_graph
async def test_node_degree(nebula_graph_storage):
    node_id = "center"
    node_data = {"entity_name": "Entity 1", "description": "description111"}
    edge_data = {"weight": 1.0, "description": "connection"}
    await nebula_graph_storage.upsert_node(node_id, node_data)

    num_neighbors = 5
    for i in range(num_neighbors):
        neighbor_id = f"neighbor{i}"
        await nebula_graph_storage.upsert_node(neighbor_id, node_data)
        await nebula_graph_storage.upsert_edge(node_id, neighbor_id, edge_data)

    degree = await nebula_graph_storage.node_degree(node_id)
    assert degree == num_neighbors

    non_existent_degree = await nebula_graph_storage.node_degree("non_existent")
    assert non_existent_degree == 0

 

@pytest.mark.asyncio
@reset_graph
async def test_edge_degree(nebula_graph_storage):
    source_id = "node1"
    target_id = "node2"
    node_data = {"entity_name": "Entity 1", "description": "description111"}
    edge_data = {"weight": 1.0, "description": "connection"}

    await nebula_graph_storage.upsert_node(source_id, node_data)
    await nebula_graph_storage.upsert_node(target_id, node_data)
    await nebula_graph_storage.upsert_edge(source_id, target_id, edge_data)

    num_source_neighbors = 3
    for i in range(num_source_neighbors):
        neighbor_id = f"neighbor{i}"
        await nebula_graph_storage.upsert_node(neighbor_id, node_data)
        await nebula_graph_storage.upsert_edge(source_id, neighbor_id, edge_data)

    num_target_neighbors = 2
    for i in range(num_target_neighbors):
        neighbor_id = f"target_neighbor{i}"
        await nebula_graph_storage.upsert_node(neighbor_id, node_data)
        await nebula_graph_storage.upsert_edge(target_id, neighbor_id, edge_data)

    expected_edge_degree = (num_source_neighbors + 1) + (num_target_neighbors + 1)
    edge_degree = await nebula_graph_storage.edge_degree(source_id, target_id)
    assert edge_degree == expected_edge_degree

    non_existent_edge_degree = await nebula_graph_storage.edge_degree("non_existent1", "non_existent2")
    assert non_existent_edge_degree == 0

    
 

@pytest.mark.asyncio
@reset_graph
async def test_get_node_edges(nebula_graph_storage):
    center_id = "center"
    await nebula_graph_storage.upsert_node(center_id, {"entity_name": "Center Node"})

    expected_edges = []
    for i in range(3):
        neighbor_id = f"neighbor{i}"
        await nebula_graph_storage.upsert_node(neighbor_id, {"entity_name": f"Neighbor {i}"})
        await nebula_graph_storage.upsert_edge(center_id, neighbor_id, {"weight": 1.0, "description": "connection"})
        expected_edges.append((center_id, neighbor_id))

    result = await nebula_graph_storage.get_node_edges(center_id)
    assert all(any(edge[0] == r[0] and edge[1] == r[1] for r in result) for edge in expected_edges)

 
@pytest.mark.parametrize("algorithm", ["leiden"])
@pytest.mark.asyncio
@reset_graph
async def test_clustering(nebula_graph_storage, algorithm):
    for i in range(10):
        await nebula_graph_storage.upsert_node(f"NODE{i}", {"source_id": f"chunk{i}"})

    for i in range(9):
        await nebula_graph_storage.upsert_edge(f"NODE{i}", f"NODE{i+1}", {"weight": 1.0})

    await nebula_graph_storage.clustering(algorithm=algorithm)

    community_schema = await nebula_graph_storage.community_schema()

    assert len(community_schema) > 0

    for community in community_schema.values():
        assert "level" in community
        assert "title" in community
        assert "edges" in community
        assert "nodes" in community
        assert "chunk_ids" in community
        assert "occurrence" in community
        assert "sub_communities" in community

    all_nodes = set()
    for community in community_schema.values():
        all_nodes.update(community["nodes"])
    assert len(all_nodes) == 10

 
@pytest.mark.parametrize("algorithm", ["leiden"])
@pytest.mark.asyncio
@reset_graph
async def test_leiden_clustering_community_structure(nebula_graph_storage, algorithm):
    for i in range(10):
        await nebula_graph_storage.upsert_node(f"A{i}", {"source_id": f"chunkA{i}"})
        await nebula_graph_storage.upsert_node(f"B{i}", {"source_id": f"chunkB{i}"})
    for i in range(9):
        await nebula_graph_storage.upsert_edge(f"A{i}", f"A{i+1}", {"weight": 1.0})
        await nebula_graph_storage.upsert_edge(f"B{i}", f"B{i+1}", {"weight": 1.0})
    
    await nebula_graph_storage.clustering(algorithm=algorithm)
    community_schema = await nebula_graph_storage.community_schema()
    
    assert len(community_schema) >= 2, "Should have at least two communities"
    
    communities = list(community_schema.values())
    a_nodes = set(node for node in communities[0]['nodes'] if node.startswith('A'))
    b_nodes = set(node for node in communities[0]['nodes'] if node.startswith('B'))
    assert len(a_nodes) == 0 or len(b_nodes) == 0, "Nodes from different groups should be in different communities"

 
@pytest.mark.parametrize("algorithm", ["leiden"])
@pytest.mark.asyncio
@reset_graph
async def test_leiden_clustering_hierarchical_structure(nebula_graph_storage, algorithm):
    await nebula_graph_storage.upsert_node("NODE1", {"source_id": "chunk1", "clusters": json.dumps([{"level": 0, "cluster": "0"}, {"level": 1, "cluster": "1"}])})
    await nebula_graph_storage.upsert_node("NODE2", {"source_id": "chunk2", "clusters": json.dumps([{"level": 0, "cluster": "0"}, {"level": 1, "cluster": "2"}])})
    await nebula_graph_storage.upsert_edge("NODE1", "NODE2", {"weight": 1.0})
    
    await nebula_graph_storage.clustering(algorithm=algorithm)
    community_schema = await nebula_graph_storage.community_schema()
    
    levels = set(community['level'] for community in community_schema.values())
    assert len(levels) >= 1, "Should have at least one level in the hierarchy"
    
    communities_per_level = {level: sum(1 for c in community_schema.values() if c['level'] == level) for level in levels}
    assert communities_per_level[0] >= communities_per_level.get(max(levels), 0), "Lower levels should have more or equal number of communities"
    

@pytest.mark.asyncio
@reset_graph
async def test_error_handling(nebula_graph_storage):
    with pytest.raises(
        ValueError, match="Clustering algorithm invalid_algo not supported"
    ):
        await nebula_graph_storage.clustering("invalid_algo")

@pytest.mark.asyncio
@reset_graph
async def test_index_done(nebula_graph_storage):
    await nebula_graph_storage.index_done_callback()
