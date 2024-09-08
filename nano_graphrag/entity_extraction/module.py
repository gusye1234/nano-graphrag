import json
import re
import dspy
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import clean_str
from nano_graphrag.entity_extraction.signature import (
    EntityTypeExtraction, 
    EntityExtraction,
    RelationshipExtraction, 
)


class EntityRelationshipExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.type_extractor = dspy.ChainOfThought(EntityTypeExtraction)
        self.entity_extractor = dspy.ChainOfThought(EntityExtraction)
        self.relationship_extractor = dspy.ChainOfThought(RelationshipExtraction)
        self.prompt_template = PROMPTS["entity_extraction"]
        self.context_base = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        )

    def forward(self, input_text: str, chunk_key: str) -> tuple[list[dict], list[dict]]:
        type_result = self.type_extractor(input_text=input_text)
        formatted_prompt = self.prompt_template.format(
            input_text=input_text,
            entity_types=type_result.entity_types,
            **self.context_base
        )
        entity_result = self.entity_extractor(input_text=formatted_prompt, entity_types=type_result.entity_types)
        relationship_result = self.relationship_extractor(input_text=formatted_prompt, entities=entity_result.entities)
        parsed_entities = self.handle_single_entity_extraction(entity_result.entities, chunk_key)
        parsed_relationships = self.handle_single_relationship_extraction(relationship_result.relationships, chunk_key)
        return parsed_entities, parsed_relationships

    def handle_single_entity_extraction(self, entities: str, chunk_key: str) -> list[dict]:
        entities = re.sub(r'^\d+\.\s*', '', entities, flags=re.MULTILINE)
        entities = entities.replace(PROMPTS["DEFAULT_COMPLETION_DELIMITER"], '').strip()
        entity_strings = re.findall(r'\{[^}]+\}', entities)
        extracted_entities = []

        for entity_str in entity_strings:   
            entity = json.loads(entity_str)
            extracted_entities.append({
                "source_id": chunk_key,
                "entity_name": clean_str(entity["name"].upper()),
                "entity_type": clean_str(entity["type"].upper()),
                "description": clean_str(entity["description"]),
                "importance_score": float(entity["importance_score"]),
            })

        return extracted_entities

    def handle_single_relationship_extraction(self, relationships: str, chunk_key: str) -> list[dict]:
        relationships = re.sub(r'^\d+\.\s*', '', relationships, flags=re.MULTILINE)
        relationships = relationships.replace(PROMPTS["DEFAULT_COMPLETION_DELIMITER"], '').strip()
        relationship_strings = re.findall(r'\{[^}]+\}', relationships)
        extracted_relationships = []
        
        for relationship_str in relationship_strings:
            relationship = json.loads(relationship_str)
            extracted_relationships.append({
                "source_id": chunk_key,
                "src_id": clean_str(relationship["source"].upper()),
                "tgt_id": clean_str(relationship["target"].upper()),
                "description": clean_str(relationship["description"]),
                "weight": float(relationship["importance_score"]),
            })

        return extracted_relationships
