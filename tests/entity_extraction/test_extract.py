import pytest
import dspy
from openai import BadRequestError
from unittest.mock import Mock, patch, AsyncMock
from nano_graphrag.entity_extraction.extract import generate_dataset, extract_entities_dspy
from nano_graphrag.base import TextChunkSchema, BaseGraphStorage, BaseVectorStorage
import httpx


@pytest.fixture
def mock_chunks():
    return {
        "chunk1": TextChunkSchema(content="Apple announced a new iPhone model."),
        "chunk2": TextChunkSchema(content="Google released an update for Android.")
    }


@pytest.fixture
def mock_entity_extractor():
    with patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock:
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
        "entity_relationship_module_path": "path/to/module.json"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_compiled,save_dataset", [
    (True, True), (False, True), (True, False), (False, False)
])
async def test_generate_dataset(mock_chunks, mock_entity_extractor, tmp_path, use_compiled, save_dataset):
    mock_prediction = Mock(
        entities=[{"entity_name": "APPLE", "entity_type": "ORGANIZATION"}],
        relationships=[{"src_id": "APPLE", "tgt_id": "IPHONE"}]
    )
    mock_entity_extractor.return_value = mock_prediction

    filepath = tmp_path / "test_dataset.pkl"

    mock_global_config = {
        "use_compiled_dspy_entity_relationship": use_compiled,
        "entity_relationship_module_path": "test/path.json" if use_compiled else None
    }

    with patch('nano_graphrag.entity_extraction.extract.pickle.dump') as mock_dump, \
         patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock_extractor_class:

        mock_extractor_instance = Mock()
        mock_extractor_instance.return_value = mock_prediction
        mock_extractor_class.return_value = mock_extractor_instance

        if use_compiled:
            mock_extractor_instance.load = Mock()

        result = await generate_dataset(chunks=mock_chunks, filepath=str(filepath), 
                                        save_dataset=save_dataset, global_config=mock_global_config)

    assert len(result) == 2
    assert isinstance(result[0], dspy.Example)
    assert hasattr(result[0], 'input_text')
    assert hasattr(result[0], 'entities')
    assert hasattr(result[0], 'relationships')

    if save_dataset:
        mock_dump.assert_called_once()
    else:
        mock_dump.assert_not_called()

    mock_extractor_class.assert_called_once()
    assert mock_extractor_instance.call_count == len(mock_chunks)

    if use_compiled:
        mock_extractor_instance.load.assert_called_once_with("test/path.json")
    else:
        assert not hasattr(mock_extractor_instance, 'load') or not mock_extractor_instance.load.called


@pytest.mark.asyncio
async def test_generate_dataset_with_empty_chunks():
    chunks = {}
    filepath = "test_empty_dataset.pkl"
    result = await generate_dataset(chunks, filepath, save_dataset=False)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_generate_dataset_with_bad_request_error():
    chunks = {"chunk1": TextChunkSchema(content="Test content")}
    filepath = "test_error_dataset.pkl"
    
    # Create a mock response object
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.headers = {"x-request-id": "test-request-id"}
    mock_response.request = Mock(spec=httpx.Request)

    with patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock_extractor_class:
        mock_extractor_instance = Mock()
        mock_extractor_instance.side_effect = BadRequestError(
            message="Test Error",
            response=mock_response,
            body={"error": {"message": "Test Error", "type": "invalid_request_error"}}
        )
        mock_extractor_class.return_value = mock_extractor_instance
        
        with patch('nano_graphrag.entity_extraction.extract.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = BadRequestError(
                message="Test Error",
                response=mock_response,
                body={"error": {"message": "Test Error", "type": "invalid_request_error"}}
            )
            
            result = await generate_dataset(chunks, filepath, save_dataset=False)
    
    assert len(result) == 0
    mock_to_thread.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_compiled,entity_vdb", [
    (True, Mock(spec=BaseVectorStorage)), 
    (False, Mock(spec=BaseVectorStorage)),
    (True, None),
    (False, None)
])
async def test_extract_entities_dspy(mock_chunks, mock_graph_storage, entity_vdb, mock_global_config, use_compiled):
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
        entities=[mock_entity],
        relationships=[mock_relationship]
    )

    mock_global_config.update({
        "use_compiled_dspy_entity_relationship": use_compiled,
        "entity_relationship_module_path": "test/path.json" if use_compiled else None
    })

    with patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock_extractor_class:
        mock_extractor_instance = Mock()
        mock_extractor_instance.return_value = mock_prediction
        mock_extractor_class.return_value = mock_extractor_instance

        if use_compiled:
            mock_extractor_instance.load = Mock()

        with patch('nano_graphrag.entity_extraction.extract._merge_nodes_then_upsert', new_callable=AsyncMock) as mock_merge_nodes, \
             patch('nano_graphrag.entity_extraction.extract._merge_edges_then_upsert', new_callable=AsyncMock) as mock_merge_edges:
            mock_merge_nodes.return_value = mock_entity
            result = await extract_entities_dspy(mock_chunks, mock_graph_storage, entity_vdb, mock_global_config)

    assert result == mock_graph_storage
    mock_extractor_class.assert_called_once()
    mock_extractor_instance.assert_called()
    mock_merge_nodes.assert_called()
    mock_merge_edges.assert_called()
    
    if entity_vdb:
        entity_vdb.upsert.assert_called_once()
    else:
        assert not hasattr(entity_vdb, 'upsert') or not entity_vdb.upsert.called

    assert mock_extractor_instance.call_count == len(mock_chunks)

    if use_compiled:
        mock_extractor_instance.load.assert_called_once_with("test/path.json")
    else:
        assert not hasattr(mock_extractor_instance, 'load') or not mock_extractor_instance.load.called


@pytest.mark.asyncio
async def test_extract_entities_dspy_with_empty_chunks():
    chunks = {}
    mock_graph_storage = Mock(spec=BaseGraphStorage)
    mock_vector_storage = Mock(spec=BaseVectorStorage)
    global_config = {}
    
    result = await extract_entities_dspy(chunks, mock_graph_storage, mock_vector_storage, global_config)
    
    assert result is None


@pytest.mark.asyncio
async def test_extract_entities_dspy_with_no_entities():
    chunks = {"chunk1": TextChunkSchema(content="Test content")}
    mock_graph_storage = Mock(spec=BaseGraphStorage)
    mock_vector_storage = Mock(spec=BaseVectorStorage)
    global_config = {}
    
    with patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock_extractor:
        mock_extractor.return_value.return_value = Mock(entities=[], relationships=[])
        result = await extract_entities_dspy(chunks, mock_graph_storage, mock_vector_storage, global_config)
    
    assert result is None
    mock_vector_storage.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_extract_entities_dspy_with_bad_request_error():
    chunks = {"chunk1": TextChunkSchema(content="Test content")}
    mock_graph_storage = Mock(spec=BaseGraphStorage)
    mock_vector_storage = Mock(spec=BaseVectorStorage)
    global_config = {}

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.headers = {"x-request-id": "test-request-id"}
    mock_response.request = Mock(spec=httpx.Request)

    with patch('nano_graphrag.entity_extraction.extract.TypedEntityRelationshipExtractor') as mock_extractor_class:
        mock_extractor_instance = Mock()
        mock_extractor_instance.side_effect = BadRequestError(
            message="Test Error",
            response=mock_response,
            body={"error": {"message": "Test Error", "type": "invalid_request_error"}}
        )
        mock_extractor_class.return_value = mock_extractor_instance
        
        with patch('nano_graphrag.entity_extraction.extract.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = BadRequestError(
                message="Test Error",
                response=mock_response,
                body={"error": {"message": "Test Error", "type": "invalid_request_error"}}
            )
            
            result = await extract_entities_dspy(chunks, mock_graph_storage, mock_vector_storage, global_config)

    assert result is None
    mock_to_thread.assert_called_once()
    mock_vector_storage.upsert.assert_not_called()
