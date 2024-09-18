import json
from unittest.mock import Mock, patch
from nano_graphrag.entity_extraction.module import EntityRelationshipExtractor


def test_entity_relationship_extractor():
    with patch('nano_graphrag.entity_extraction.module.dspy.ChainOfThought') as mock_chain_of_thought:
        input_text = "Apple announced a new iPhone model."
        mock_extractor = Mock()
        mock_self_reflection = Mock()
        mock_chain_of_thought.side_effect = [mock_extractor, mock_self_reflection]

        mock_entities = [
            {"entity_name": "APPLE", "entity_type": "ORGANIZATION", "description": "A technology company", "importance_score": 1},
            {"entity_name": "IPHONE", "entity_type": "PRODUCT", "description": "A smartphone", "importance_score": 1}
        ]
        mock_relationships = [
            {"src_id": "APPLE", "tgt_id": "IPHONE", "description": "Apple manufactures iPhone", "weight": 1, "order": 1}
        ]
        mock_missing_entities = [
            {"entity_name": "TIM_COOK", "entity_type": "PERSON", "description": "CEO of Apple", "importance_score": 0.8}
        ]
        mock_missing_relationships = [
            {"src_id": "TIM_COOK", "tgt_id": "APPLE", "description": "Tim Cook is the CEO of Apple", "weight": 0.9, "order": 1},
            {"src_id": "APPLE", "tgt_id": "IPHONE", "description": "Apple announces new iPhone model", "weight": 1, "order": 1}
        ]

        mock_extractor.return_value = Mock(
            entities=json.dumps(mock_entities),
            relationships=json.dumps(mock_relationships)
        )
        mock_self_reflection.return_value = Mock(
            missing_entities=json.dumps(mock_missing_entities),
            missing_relationships=json.dumps(mock_missing_relationships)
        )

        extractor = EntityRelationshipExtractor()
        result = extractor.forward(input_text=input_text)

        mock_extractor.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types.model_dump_json()
        )

        mock_self_reflection.assert_called_once_with(
            input_text=input_text,
            entity_types=extractor.entity_types.model_dump_json(),
            entities=mock_extractor.return_value.entities,
            relationships=mock_extractor.return_value.relationships
        )

        assert len(result.entities) == 3
        assert len(result.relationships) == 3

        assert result.entities[0]["entity_name"] == "APPLE"
        assert result.entities[0]["entity_type"] == "ORGANIZATION"
        assert result.entities[0]["description"] == "A technology company"

        assert result.entities[1]["entity_name"] == "IPHONE"
        assert result.entities[1]["entity_type"] == "PRODUCT"
        assert result.entities[1]["description"] == "A smartphone"
        assert result.entities[1]["importance_score"] == 1

        assert result.entities[2]["entity_name"] == "TIM_COOK"
        assert result.entities[2]["entity_type"] == "PERSON"
        assert result.entities[2]["description"] == "CEO of Apple"
        assert result.entities[2]["importance_score"] == 0.8

        assert result.relationships[0]["src_id"] == "APPLE"
        assert result.relationships[0]["tgt_id"] == "IPHONE"
        assert result.relationships[0]["description"] == "Apple manufactures iPhone"
        assert result.relationships[0]["weight"] == 1
        assert result.relationships[0]["order"] == 1

        assert result.relationships[1]["src_id"] == "TIM_COOK"
        assert result.relationships[1]["tgt_id"] == "APPLE"
        assert result.relationships[1]["description"] == "Tim Cook is the CEO of Apple"
        assert result.relationships[1]["weight"] == 0.9
        assert result.relationships[1]["order"] == 1

        assert result.relationships[2]["src_id"] == "APPLE"
        assert result.relationships[2]["tgt_id"] == "IPHONE"
        assert result.relationships[2]["description"] == "Apple announces new iPhone model"
        assert result.relationships[2]["weight"] == 1
        assert result.relationships[2]["order"] == 1
