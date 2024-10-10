import pytest
import dspy
from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.module import TypedEntityRelationshipExtractor, Relationship, Entity


@pytest.mark.parametrize("self_refine,num_refine_turns", [
    (False, 0),
    (True, 2)
])
def test_entity_relationship_extractor(self_refine, num_refine_turns):
    with patch('nano_graphrag.entity_extraction.module.dspy.TypedChainOfThought') as mock_chain_of_thought:
        input_text = "Apple announced a new iPhone model."
        mock_extractor = Mock()
        mock_critique = Mock()
        mock_refine = Mock()
        
        mock_chain_of_thought.side_effect = [mock_extractor, mock_critique, mock_refine]

        mock_entities = [
            Entity(entity_name="APPLE", entity_type="ORGANIZATION", description="A technology company", importance_score=1),
            Entity(entity_name="IPHONE", entity_type="PRODUCT", description="A smartphone", importance_score=1)
        ]
        mock_relationships = [
            Relationship(src_id="APPLE", tgt_id="IPHONE", description="Apple manufactures iPhone", weight=1, order=1)
        ]

        mock_extractor.return_value = dspy.Prediction(
            entities=mock_entities, relationships=mock_relationships
        )
        
        if self_refine:
            mock_critique.return_value = dspy.Prediction(
                entity_critique="Good entities, but could be more detailed.",
                relationship_critique="Relationships are accurate but limited."
            )
            mock_refine.return_value = dspy.Prediction(
                refined_entities=mock_entities,
                refined_relationships=mock_relationships
            )
        
        extractor = TypedEntityRelationshipExtractor(self_refine=self_refine, num_refine_turns=num_refine_turns)
        result = extractor.forward(input_text=input_text)

        mock_extractor.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types
        )

        if self_refine:
            assert mock_critique.call_count == num_refine_turns
            assert mock_refine.call_count == num_refine_turns

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
