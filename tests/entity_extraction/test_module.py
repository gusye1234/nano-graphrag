import dspy
from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.module import TypedEntityRelationshipExtractor, Relationship, Entity


def test_entity_relationship_extractor():
    with patch('nano_graphrag.entity_extraction.module.dspy.TypedChainOfThought') as mock_chain_of_thought:
        input_text = "Apple announced a new iPhone model."
        mock_extractor = Mock()
        mock_chain_of_thought.return_value = mock_extractor

        mock_entities = [
            Entity(entity_name="APPLE", entity_type="ORGANIZATION", description="A technology company", importance_score=1),
            Entity(entity_name="IPHONE", entity_type="PRODUCT", description="A smartphone", importance_score=1)
        ]
        mock_relationships = [
            Relationship(src_id="APPLE", tgt_id="IPHONE", description="Apple manufactures iPhone", weight=1, order=1)
        ]

        mock_extractor.return_value = dspy.Prediction(
            entities_relationships=mock_entities + mock_relationships
        )
        
        extractor = TypedEntityRelationshipExtractor()
        result = extractor.forward(input_text=input_text)

        mock_extractor.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types
        )

        assert len(result.entities) == 2
        assert len(result.relationships) == 1

        assert result.entities[0]["entity_name"] == "APPLE"
        assert result.entities[0]["entity_type"] == "ORGANIZATION"
        assert result.entities[0]["description"] == "A technology company"
        assert result.entities[0]["importance_score"] == 1

        assert result.entities[1]["entity_name"] == "IPHONE"
        assert result.entities[1]["entity_type"] == "PRODUCT"
        assert result.entities[1]["description"] == "A smartphone"
        assert result.entities[1]["importance_score"] == 1

        assert result.relationships[0]["src_id"] == "APPLE"
        assert result.relationships[0]["tgt_id"] == "IPHONE"
        assert result.relationships[0]["description"] == "Apple manufactures iPhone"
        assert result.relationships[0]["weight"] == 1
        assert result.relationships[0]["order"] == 1
