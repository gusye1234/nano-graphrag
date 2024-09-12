from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.module import (
    EntityRelationshipExtractor,
    Entities,
    Relationships,
    Entity,
    Relationship
)


def test_entity_relationship_extractor():
    with patch('nano_graphrag.entity_extraction.module.dspy.TypedPredictor') as mock_typed_predictor:
        input_text = "Apple announced a new iPhone model."
        mock_extractor = Mock()
        mock_self_reflection = Mock()
        mock_typed_predictor.side_effect = [mock_extractor, mock_self_reflection]

        mock_entities = [
            Entity(entity_name="APPLE", entity_type="ORGANIZATION", description="A technology company", importance_score=1),
            Entity(entity_name="IPHONE", entity_type="PRODUCT", description="A smartphone", importance_score=1)
        ]
        mock_relationships = [
            Relationship(src_id="APPLE", tgt_id="IPHONE", description="Apple manufactures iPhone", weight=1, order=1)
        ]
        mock_missing_entities = [
            Entity(entity_name="TIM_COOK", entity_type="PERSON", description="CEO of Apple", importance_score=0.8)
        ]
        mock_missing_relationships = [
            Relationship(src_id="TIM_COOK", tgt_id="APPLE", description="Tim Cook is the CEO of Apple", weight=0.9, order=1),
            Relationship(src_id="APPLE", tgt_id="IPHONE", description="Apple announces new iPhone model", weight=1, order=1)
        ]

        mock_extractor.return_value = Mock(
            entities=Entities(context=mock_entities),
            relationships=Relationships(context=mock_relationships)
        )
        mock_self_reflection.return_value = Mock(
            missing_entities=Entities(context=mock_missing_entities),
            missing_relationships=Relationships(context=mock_missing_relationships)
        )

        extractor = EntityRelationshipExtractor()
        result = extractor.forward(input_text=input_text)

        mock_extractor.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types
        )

        mock_self_reflection.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types,
            entities=mock_extractor.return_value.entities,
            relationships=mock_extractor.return_value.relationships
        )

        assert len(result.entities.context) == 3
        assert len(result.relationships.context) == 3

        assert result.entities.context[0].entity_name == "APPLE"
        assert result.entities.context[0].entity_type == "ORGANIZATION"
        assert result.entities.context[0].description == "A technology company"

        assert result.entities.context[1].entity_name == "IPHONE"
        assert result.entities.context[1].entity_type == "PRODUCT"
        assert result.entities.context[1].description == "A smartphone"
        assert result.entities.context[1].importance_score == 1

        assert result.entities.context[2].entity_name == "TIM_COOK"
        assert result.entities.context[2].entity_type == "PERSON"
        assert result.entities.context[2].description == "CEO of Apple"
        assert result.entities.context[2].importance_score == 0.8

        assert result.relationships.context[0].src_id == "APPLE"
        assert result.relationships.context[0].tgt_id == "IPHONE"
        assert result.relationships.context[0].description == "Apple manufactures iPhone"
        assert result.relationships.context[0].weight == 1
        assert result.relationships.context[0].order == 1

        assert result.relationships.context[1].src_id == "TIM_COOK"
        assert result.relationships.context[1].tgt_id == "APPLE"
        assert result.relationships.context[1].description == "Tim Cook is the CEO of Apple"
        assert result.relationships.context[1].weight == 0.9
        assert result.relationships.context[1].order == 1

        assert result.relationships.context[2].src_id == "APPLE"
        assert result.relationships.context[2].tgt_id == "IPHONE"
        assert result.relationships.context[2].description == "Apple announces new iPhone model"
        assert result.relationships.context[2].weight == 1
        assert result.relationships.context[2].order == 1
