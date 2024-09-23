import pytest
import dspy
from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.metric import (
    relationships_similarity_metric,
    entity_recall_metric,
)


@pytest.fixture
def mock_dspy_predict():
    with patch('nano_graphrag.entity_extraction.metric.dspy.TypedChainOfThought') as mock_predict:
        mock_instance = Mock()
        mock_instance.return_value = dspy.Prediction(similarity_score=0.75)
        mock_predict.return_value = mock_instance
        yield mock_predict


@pytest.fixture
def sample_relationship():
    return {
        "src_id": "ENTITY1",
        "tgt_id": "ENTITY2",
        "description": "Example relationship",
        "weight": 0.8,
        "order": 1
    }


@pytest.fixture
def sample_entity():
    return {
        "entity_name": "EXAMPLE_ENTITY",
        "entity_type": "PERSON",
        "description": "An example entity",
        "importance_score": 0.8
    }


@pytest.fixture
def example():
    def _example(items):
        return {"relationships": items} if "src_id" in (items[0] if items else {}) else {"entities": items}
    return _example


@pytest.fixture
def prediction():
    def _prediction(items):
        return {"relationships": items} if "src_id" in (items[0] if items else {}) else {"entities": items}
    return _prediction


@pytest.mark.asyncio
async def test_relationship_similarity_metric(sample_relationship, example, prediction, mock_dspy_predict):
    gold = example([
        {**sample_relationship, "src_id": "ENTITY1", "tgt_id": "ENTITY2", "description": "is related to"},
        {**sample_relationship, "src_id": "ENTITY2", "tgt_id": "ENTITY3", "description": "is connected with"},
    ])
    pred = prediction([
        {**sample_relationship, "src_id": "ENTITY1", "tgt_id": "ENTITY2", "description": "is connected to"},
        {**sample_relationship, "src_id": "ENTITY2", "tgt_id": "ENTITY3", "description": "is linked with"},
    ])

    similarity = relationships_similarity_metric(gold, pred)
    assert 0 <= similarity <= 1


@pytest.mark.asyncio
async def test_entity_recall_metric(sample_entity, example, prediction):
    gold = example([
        {**sample_entity, "entity_name": "ENTITY1"},
        {**sample_entity, "entity_name": "ENTITY2"},
        {**sample_entity, "entity_name": "ENTITY3"},
    ])
    pred = example([
        {**sample_entity, "entity_name": "ENTITY1"},
        {**sample_entity, "entity_name": "ENTITY3"},
        {**sample_entity, "entity_name": "ENTITY4"},
    ])

    recall = entity_recall_metric(gold, pred)
    assert recall == 2/3


@pytest.mark.asyncio
async def test_relationship_similarity_metric_no_common_keys(sample_relationship, example, prediction, mock_dspy_predict):
    gold = example([{**sample_relationship, "src_id": "ENTITY1", "tgt_id": "ENTITY2", "description": "is related to"}])
    pred = prediction([{**sample_relationship, "src_id": "ENTITY3", "tgt_id": "ENTITY4", "description": "is connected with"}])

    similarity = relationships_similarity_metric(gold, pred)
    assert 0 <= similarity <= 1


@pytest.mark.asyncio
async def test_entity_recall_metric_no_true_positives(sample_entity, example, prediction):
    gold = example([
        {**sample_entity, "entity_name": "ENTITY1"},
        {**sample_entity, "entity_name": "ENTITY2"}
    ])
    pred = prediction([
        {**sample_entity, "entity_name": "ENTITY3"},
        {**sample_entity, "entity_name": "ENTITY4"}
    ])

    recall = entity_recall_metric(gold, pred)
    assert recall == 0



@pytest.mark.asyncio
async def test_relationship_similarity_metric_identical_descriptions(sample_relationship, example, prediction, mock_dspy_predict):
    gold = example([{**sample_relationship, "src_id": "ENTITY1", "tgt_id": "ENTITY2", "description": "is related to"}])
    pred = prediction([{**sample_relationship, "src_id": "ENTITY1", "tgt_id": "ENTITY2", "description": "is related to"}])

    similarity = relationships_similarity_metric(gold, pred)
    assert similarity == 0.75


@pytest.mark.asyncio
async def test_entity_recall_metric_perfect_recall(sample_entity, example, prediction):
    entities = [
        {**sample_entity, "entity_name": "ENTITY1"},
        {**sample_entity, "entity_name": "ENTITY2"}
    ]
    gold = example(entities)
    pred = prediction(entities)

    recall = entity_recall_metric(gold, pred)
    assert recall == 1.0


@pytest.mark.asyncio
async def test_relationship_similarity_metric_no_relationships(example, prediction, mock_dspy_predict):
    gold = example([])
    pred = prediction([])

    with pytest.raises(KeyError):
        similarity = relationships_similarity_metric(gold, pred)
