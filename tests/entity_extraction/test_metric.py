import pytest
import numpy as np
from nano_graphrag.entity_extraction.metric import (
    local_embedding,
    relationship_similarity_metric,
    relationship_recall_metric,
    entity_recall_metric,
    EMBED_MODEL
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


@pytest.mark.asyncio
async def test_local_embedding(sample_texts):
    embeddings = local_embedding(sample_texts)
    assert embeddings.shape == (3, 384)
    assert isinstance(embeddings, np.ndarray)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


@pytest.mark.asyncio
async def test_relationship_similarity_metric(relationship, example, prediction):
    gold = example([
        relationship("1", "2", "is related to"),
        relationship("2", "3", "is connected with"),
    ])
    pred = prediction([
        relationship("1", "2", "is connected to"),
        relationship("2", "3", "is linked with"),
    ])

    similarity = relationship_similarity_metric(gold, pred)
    assert 0 <= similarity <= 1


@pytest.mark.asyncio
async def test_relationship_recall_metric(relationship, example, prediction):
    gold = example([
        relationship("1", "2"),
        relationship("2", "3"),
        relationship("3", "4"),
    ])
    pred = prediction([
        relationship("1", "2"),
        relationship("2", "3"),
    ])

    recall = relationship_recall_metric(gold, pred)
    assert recall == 2/3


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
async def test_relationship_similarity_metric_no_common_keys(relationship, example, prediction):
    gold = example([relationship("1", "2", "is related to")])
    pred = prediction([relationship("3", "4", "is connected with")])

    similarity = relationship_similarity_metric(gold, pred)
    assert similarity == 0.0


@pytest.mark.asyncio
async def test_relationship_recall_metric_no_true_positives(relationship, example, prediction):
    gold = example([relationship("1", "2"), relationship("2", "3")])
    pred = prediction([relationship("3", "4"), relationship("4", "5")])

    recall = relationship_recall_metric(gold, pred)
    assert recall == 0


@pytest.mark.asyncio
async def test_entity_recall_metric_no_true_positives(entity, example, prediction):
    gold = example([entity("Entity1"), entity("Entity2")])
    pred = prediction([entity("Entity3"), entity("Entity4")])

    recall = entity_recall_metric(gold, pred)
    assert recall == 0


@pytest.mark.asyncio
async def test_relationship_similarity_metric_identical_descriptions(relationship, example, prediction):
    gold = example([relationship("1", "2", "is related to")])
    pred = prediction([relationship("1", "2", "is related to")])

    similarity = relationship_similarity_metric(gold, pred)
    assert np.isclose(similarity, 1.0, atol=1e-6)


@pytest.mark.asyncio
async def test_relationship_recall_metric_perfect_recall(relationship, example, prediction):
    relationships = [relationship("1", "2"), relationship("2", "3")]
    gold = example(relationships)
    pred = prediction(relationships)

    recall = relationship_recall_metric(gold, pred)
    assert recall == 1.0


@pytest.mark.asyncio
async def test_entity_recall_metric_perfect_recall(entity, example, prediction):
    entities = [entity("Entity1"), entity("Entity2")]
    gold = example(entities)
    pred = prediction(entities)

    recall = entity_recall_metric(gold, pred)
    assert recall == 1.0


@pytest.mark.asyncio
async def test_embed_model_consistency(sample_texts):
    embeddings1 = EMBED_MODEL.encode(sample_texts, normalize_embeddings=True)
    embeddings2 = local_embedding(sample_texts)
    assert np.allclose(embeddings1, embeddings2)
