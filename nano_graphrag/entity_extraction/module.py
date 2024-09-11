import dspy
from nano_graphrag._utils import logger
from nano_graphrag.entity_extraction.signature import (
    EntityTypes,
    Entities,
    Relationships,
    CombinedExtraction,
    CombinedSelfReflection
)


class EntityRelationshipExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.entity_types = EntityTypes()
        self.extractor = dspy.TypedPredictor(CombinedExtraction)
        self.self_reflection = dspy.TypedPredictor(CombinedSelfReflection)

    def forward(self, input_text: str) -> dspy.Prediction:
        extraction_result = self.extractor(input_text=input_text, entity_types=self.entity_types)
        reflection_result = self.self_reflection(
            input_text=input_text,
            entity_types=self.entity_types,
            entities=extraction_result.entities,
            relationships=extraction_result.relationships
        )
        entities = extraction_result.entities
        missing_entities = reflection_result.missing_entities
        relationships = extraction_result.relationships
        missing_relationships = reflection_result.missing_relationships
        parsed_entities = Entities(context=entities.context + missing_entities.context)
        parsed_relationships = Relationships(context=relationships.context + missing_relationships.context)
        logger.debug(f"Entities: {len(entities.context)} | Missed Entities: {len(missing_entities.context)} | Total Entities: {len(parsed_entities.context)}")
        logger.debug(f"Relationships: {len(relationships.context)} | Missed Relationships: {len(missing_relationships.context)} | Total Relationships: {len(parsed_relationships.context)}")
        return dspy.Prediction(entities=parsed_entities, relationships=parsed_relationships)
    