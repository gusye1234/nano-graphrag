import pytest
import dspy
from unittest.mock import Mock, patch, AsyncMock
from nano_graphrag.entity_extraction.module import (
    Entities,
    Relationships,
)
from nano_graphrag.entity_extraction.extract import generate_dataset, extract_entities_dspy
from nano_graphrag.base import TextChunkSchema, BaseGraphStorage, BaseVectorStorage


@pytest.fixture
def mock_chunks():
    return {
        "chunk1": TextChunkSchema(content="Apple announced a new iPhone model."),
        "chunk2": TextChunkSchema(content="Google released an update for Android.")
    }


@pytest.fixture
def mock_entity_extractor():
    with patch('nano_graphrag.entity_extraction.extract.EntityRelationshipExtractor') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_graph_storage():
    return Mock(spec=BaseGraphStorage)


@pytest.fixture
def mock_vector_storage():
    return Mock(spec=BaseVectorStorage)


@pytest.fixture
def mock_global_config():
    return {
        "use_compiled_dspy_entity_relationship": False,
        "entity_relationship_module_path": "path/to/module"
    }


@pytest.mark.asyncio
async def test_generate_dataset(mock_chunks, mock_entity_extractor, tmp_path):
    mock_prediction = Mock(
        entities=Mock(context=[{"entity_name": "APPLE", "entity_type": "ORGANIZATION"}]),
        relationships=Mock(context=[{"src_id": "APPLE", "tgt_id": "IPHONE"}])
    )
    mock_entity_extractor.return_value = mock_prediction

    filepath = tmp_path / "test_dataset.pkl"
    
    with patch('nano_graphrag.entity_extraction.extract.pickle.dump') as mock_dump:
        result = await generate_dataset(mock_chunks, str(filepath))
    
    assert len(result) == 2
    assert isinstance(result[0], dspy.Example)
    assert hasattr(result[0], 'input_text')
    assert hasattr(result[0], 'entities')
    assert hasattr(result[0], 'relationships')
    assert result[0].input_text == "Apple announced a new iPhone model."
    assert result[0].entities.context == [{"entity_name": "APPLE", "entity_type": "ORGANIZATION"}]
    assert result[0].relationships.context == [{"src_id": "APPLE", "tgt_id": "IPHONE"}]


@pytest.mark.asyncio
async def test_extract_entities_dspy(mock_chunks, mock_graph_storage, mock_vector_storage, mock_global_config):
    mock_entity = {
        "entity_name": "APPLE",
        "entity_type": "ORGANIZATION",
        "description": "A tech company",
        "importance_score": 0.9
    }
    mock_relationship = {
        "src_id": "APPLE",
        "tgt_id": "IPHONE",
        "description": "Produces",
        "weight": 0.8,
        "order": 1
    }
    mock_prediction = Mock(
        entities=Entities(context=[mock_entity]),
        relationships=Relationships(context=[mock_relationship])
    )

    with patch('nano_graphrag.entity_extraction.extract.EntityRelationshipExtractor') as mock_extractor_class:
        mock_extractor_instance = Mock()
        mock_extractor_instance.return_value = mock_prediction
        mock_extractor_class.return_value = mock_extractor_instance

        with patch('nano_graphrag.entity_extraction.extract._merge_nodes_then_upsert', new_callable=AsyncMock) as mock_merge_nodes, \
             patch('nano_graphrag.entity_extraction.extract._merge_edges_then_upsert', new_callable=AsyncMock) as mock_merge_edges:
            mock_merge_nodes.return_value = mock_entity
            result = await extract_entities_dspy(mock_chunks, mock_graph_storage, mock_vector_storage, mock_global_config)

    assert result == mock_graph_storage
    mock_extractor_class.assert_called_once()
    mock_extractor_instance.assert_called()
    mock_merge_nodes.assert_called()
    mock_merge_edges.assert_called()
    mock_vector_storage.upsert.assert_called_once()
