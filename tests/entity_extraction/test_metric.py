import pytest
import numpy as np
import dspy
from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.metric import (
    relationship_similarity_metric,
    entity_recall_metric,
)


@pytest.fixture
def relationship():
    class Relationship:
        def __init__(self, src_id, tgt_id, description=None):
            self.src_id = src_id
            self.tgt_id = tgt_id
            self.description = description
    return Relationship


@pytest.fixture
def entity():
    class Entity:
        def __init__(self, entity_name):
            self.entity_name = entity_name
    return Entity


@pytest.fixture
def example():
    class Example:
        def __init__(self, items):
            self.relationships = type('obj', (object,), {'context': items})
            self.entities = type('obj', (object,), {'context': items})
    return Example


@pytest.fixture
def prediction():
    class Prediction:
        def __init__(self, items):
            self.relationships = type('obj', (object,), {'context': items})
            self.entities = type('obj', (object,), {'context': items})
    return Prediction


@pytest.fixture
def sample_texts():
    return ["Hello", "World", "Test"]


@pytest.fixture
def mock_dspy_predict():
    with patch('nano_graphrag.entity_extraction.metric.dspy.Predict') as mock_predict:
        mock_instance = Mock()
        mock_instance.return_value = dspy.Prediction(similarity_score="0.75")
        mock_predict.return_value = mock_instance
        yield mock_predict


@pytest.mark.asyncio
async def test_relationship_similarity_metric(relationship, example, prediction, mock_dspy_predict):
    gold = example([
        relationship("1", "2", "is related to"),
        relationship("2", "3", "is connected with"),
    ])
    pred = prediction([
        relationship("1", "2", "is connected to"),
        relationship("2", "3", "is linked with"),
    ])

    similarity = relationship_similarity_metric(gold, pred)
    assert np.isclose(similarity, 0.75, atol=1e-6)


@pytest.mark.asyncio
async def test_entity_recall_metric(entity, example, prediction):
    gold = example([
        entity("Entity1"),
        entity("Entity2"),
        entity("Entity3"),
    ])
    pred = prediction([
        entity("Entity1"),
        entity("Entity3"),
        entity("Entity4"),
    ])

    recall = entity_recall_metric(gold, pred)
    assert recall == 2/3


@pytest.mark.asyncio
async def test_relationship_similarity_metric_no_common_keys(relationship, example, prediction, mock_dspy_predict):
    gold = example([relationship("1", "2", "is related to")])
    pred = prediction([relationship("3", "4", "is connected with")])

    similarity = relationship_similarity_metric(gold, pred)
    assert similarity == 0.75  # The mocked value


@pytest.mark.asyncio
async def test_entity_recall_metric_no_true_positives(entity, example, prediction):
    gold = example([entity("Entity1"), entity("Entity2")])
    pred = prediction([entity("Entity3"), entity("Entity4")])

    recall = entity_recall_metric(gold, pred)
    assert recall == 0


@pytest.mark.asyncio
async def test_relationship_similarity_metric_identical_descriptions(relationship, example, prediction, mock_dspy_predict):
    gold = example([relationship("1", "2", "is related to")])
    pred = prediction([relationship("1", "2", "is related to")])

    similarity = relationship_similarity_metric(gold, pred)
    assert np.isclose(similarity, 0.75, atol=1e-6)


@pytest.mark.asyncio
async def test_entity_recall_metric_perfect_recall(entity, example, prediction):
    entities = [entity("Entity1"), entity("Entity2")]
    gold = example(entities)
    pred = prediction(entities)

    recall = entity_recall_metric(gold, pred)
    assert recall == 1.0


@pytest.mark.asyncio
async def test_relationship_similarity_metric_no_relationships(example, prediction, mock_dspy_predict):
    gold = example([])
    pred = prediction([])

    similarity = relationship_similarity_metric(gold, pred)
    assert similarity == 0.0


@pytest.mark.asyncio
async def test_relationship_similarity_metric_invalid_score(relationship, example, prediction):
    with patch('nano_graphrag.entity_extraction.metric.dspy.Predict') as mock_predict:
        mock_instance = Mock()
        mock_instance.return_value = dspy.Prediction(similarity_score="invalid")
        mock_predict.return_value = mock_instance

        gold = example([relationship("1", "2", "is related to")])
        pred = prediction([relationship("1", "2", "is connected to")])

        similarity = relationship_similarity_metric(gold, pred)
        assert similarity == 0.0
