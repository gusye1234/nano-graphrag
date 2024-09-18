import dspy
import json
from pydantic import BaseModel, Field
from nano_graphrag._utils import logger, clean_str


class EntityTypes(BaseModel):
    """
    Obtained from:
    https://github.com/SciPhi-AI/R2R/blob/6e958d1e451c1cb10b6fc868572659785d1091cb/r2r/providers/prompts/defaults.jsonl
    """
    context: list[str] = Field(
        default=[
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", 
            "PERCENTAGE", "PRODUCT", "EVENT", "LANGUAGE", "NATIONALITY", 
            "RELIGION", "TITLE", "PROFESSION", "ANIMAL", "PLANT", "DISEASE", 
            "MEDICATION", "CHEMICAL", "MATERIAL", "COLOR", "SHAPE", 
            "MEASUREMENT", "WEATHER", "NATURAL_DISASTER", "AWARD", "LAW", 
            "CRIME", "TECHNOLOGY", "SOFTWARE", "HARDWARE", "VEHICLE", 
            "FOOD", "DRINK", "SPORT", "MUSIC_GENRE", "INSTRUMENT", 
            "ARTWORK", "BOOK", "MOVIE", "TV_SHOW", "ACADEMIC_SUBJECT", 
            "SCIENTIFIC_THEORY", "POLITICAL_PARTY", "CURRENCY", 
            "STOCK_SYMBOL", "FILE_TYPE", "PROGRAMMING_LANGUAGE", 
            "MEDICAL_PROCEDURE", "CELESTIAL_BODY"
        ],
        description="List of entity types used for extraction."
    )


class CombinedExtraction(dspy.Signature):
    """Signature for extracting both entities and relationships from input text."""

    input_text = dspy.InputField(desc="The text to extract entities and relationships from.")
    entity_types = dspy.InputField()
    entities = dspy.OutputField(
        desc="""
        Format:
        {
            "entities": [
                {
                    "entity_name": "ENTITY NAME",
                    "entity_type": "ENTITY TYPE",
                    "description": "Detailed description",
                    "importance_score": "Importance score of the entity. Should be between 0 and 1 with 1 being the most important."
                }
            ]
        }
        Each entity name should be an actual atomic word from the input text. Avoid duplicates and generic terms.
        Make sure descriptions are detailed and comprehensive, including:
        1. The entity's role or significance in the context
        2. Key attributes or characteristics
        3. Relationships to other entities (if applicable)
        4. Historical or cultural relevance (if applicable)
        5. Any notable actions or events associated with the entity
        All entity types from the text must be included. 
        Entities must have an importance score greater than 0.5.
        IMPORTANT: Only use entity types from the provided 'entity_types' list. Do not introduce new entity types.
        Ensure the output is strictly JSON formatted without any trailing text or comments.
        """
    )
    relationships = dspy.OutputField(
        desc="""
        Format:
        {
            "relationships": [
                {
                    "src_id": "SOURCE ENTITY",
                    "tgt_id": "TARGET ENTITY",
                    "description": "Detailed description of the relationship",
                    "weight": "Weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
                    "order": "Order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order, etc."
                }
            ]
        }
        Make sure relationship descriptions are detailed and comprehensive, including:
        1. The nature of the relationship (e.g., familial, professional, causal)
        2. The impact or significance of the relationship on both entities
        3. Any historical or contextual information relevant to the relationship
        4. How the relationship evolved over time (if applicable)
        5. Any notable events or actions that resulted from this relationship
        Include direct relationships (order 1) as well as higher-order relationships (order 2 and 3):
        - Direct relationships: Immediate connections between entities.
        - Second-order relationships: Indirect effects or connections that result from direct relationships.
        - Third-order relationships: Further indirect effects that result from second-order relationships.
        IMPORTANT: Only include relationships between existing entities from the extracted entities. Do not introduce new entities here.
        The "src_id" and "tgt_id" fields must exactly match entity names from the extracted entities list.
        Ensure the output is strictly JSON formatted without any trailing text or comments.
        """
    )


class CombinedSelfReflection(dspy.Signature):
    """
    Signature for combined self-reflection on extracted entities and relationships.
    Self-reflection is on the completeness and quality of both the extracted entities and relationships.
    """

    input_text = dspy.InputField(desc="The original input text.")
    entity_types = dspy.InputField()
    entities = dspy.InputField(
        desc="""
        List of extracted entities.
        Format:
        {
            "entities": [
                {
                    "entity_name": "ENTITY NAME",
                    "entity_type": "ENTITY TYPE",
                    "description": "Detailed description",
                    "importance_score": "Importance score of the entity. Should be between 0 and 1 with 1 being the most important."
                }
            ]
        }
        """
    )
    relationships = dspy.InputField(
        desc="""
        List of extracted relationships.
        Format:
        {
            "relationships": [
                {
                    "src_id": "SOURCE ENTITY",
                    "tgt_id": "TARGET ENTITY",
                    "description": "Detailed description of the relationship",
                    "weight": "Weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
                    "order": "Order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order, etc."
                }
            ]
        }
        """
    )
    missing_entities  = dspy.OutputField(
        desc="""
        Format:
        {
            "entities": [
                {
                    "entity_name": "ENTITY NAME",
                    "entity_type": "ENTITY TYPE",
                    "description": "Detailed description",
                    "importance_score": "Importance score of the entity. Should be between 0 and 1 with 1 being the most important."
                }
            ]
        }
        More specifically:
        1. Entities mentioned in the text but not captured in the initial extraction.
        2. Implicit entities that are crucial to the context but not explicitly mentioned.
        3. Entities that belong to the identified entity types but were overlooked.
        4. Subtypes or more specific instances of the already extracted entities.
        Ensure the output is strictly JSON formatted without any trailing text or comments.
        """
    )
    missing_relationships = dspy.OutputField(
        desc="""
        Format:
        {
            "relationships": [
                {
                    "src_id": "SOURCE ENTITY",
                    "tgt_id": "TARGET ENTITY",
                    "description": "Detailed description of the relationship",
                    "weight": "Weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
                    "order": "Order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order, etc."
                }
            ]
        }
        More specifically:
        1. Direct relationships (order 1) between entities that were not captured initially.
        2. Second-order relationships (order 2): Indirect effects or connections resulting from direct relationships.
        3. Third-order relationships (order 3): Further indirect effects resulting from second-order relationships.
        4. Implicit relationships that can be inferred from the context.
        5. Hierarchical, causal, or temporal relationships that may have been overlooked.
        6. Relationships involving the newly identified missing entities.
        Only include relationships between entities in the combined entities list (extracted + missing).
        Ensure the output is strictly JSON formatted without any trailing text or comments.
        """
    )


class EntityRelationshipExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.entity_types = EntityTypes()
        self.extractor = dspy.ChainOfThought(CombinedExtraction)
        self.self_reflection = dspy.ChainOfThought(CombinedSelfReflection)

    def clean_json_string(self, json_string: str) -> list:
        if not json_string.strip():
            logger.warning("Received an empty JSON string")
            return []

        try:
            cleaned = json_string.strip()
            cleaned = cleaned.replace('```json', '').replace('```', '')
            cleaned = cleaned.strip()
            json_obj = json.loads(cleaned)

            if isinstance(json_obj, dict):
                for k, v in json_obj.items():
                    if isinstance(v, list) and all(isinstance(i, dict) for i in v):
                        return v 
            return json_obj if isinstance(json_obj, list) else []

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, returning empty list")
            return []
    
    def forward(self, input_text: str) -> dspy.Prediction:
        try:
            extraction_result = self.extractor(input_text=input_text, entity_types=self.entity_types.model_dump_json())
        except Exception as e:
            logger.error(f"Error in extraction: {e}")
            extraction_result = dspy.Prediction(entities='', relationships='')

        entities = self.clean_json_string(extraction_result.entities) or []
        relationships = self.clean_json_string(extraction_result.relationships) or []
        
        try:
            reflection_result = self.self_reflection(
                input_text=input_text,
                entity_types=self.entity_types.model_dump_json(),
                entities=json.dumps(entities),
                relationships=json.dumps(relationships)
            )
        except Exception as e:
            logger.warning(f"Self-reflection failed: {str(e)}")
            reflection_result = dspy.Prediction(missing_entities='', missing_relationships='')

        missing_entities = self.clean_json_string(reflection_result.missing_entities) or []
        missing_relationships = self.clean_json_string(reflection_result.missing_relationships) or []

        all_entities = entities + missing_entities
        all_relationships = relationships + missing_relationships
        
        self.log_extraction_stats(entities, missing_entities, relationships, missing_relationships)

        all_entities, all_relationships = self.clean_and_validate(all_entities, all_relationships)
        
        return dspy.Prediction(entities=all_entities, relationships=all_relationships)

    def log_extraction_stats(self, entities: list[dict], missing_entities: list[dict], relationships: list[dict], missing_relationships: list[dict]):
        logger.debug(f"Entities: {len(entities)} | Missed Entities: {len(missing_entities)} | Total Entities: {len(entities) + len(missing_entities)}")
        logger.debug(f"Relationships: {len(relationships)} | Missed Relationships: {len(missing_relationships)} | Total Relationships: {len(relationships) + len(missing_relationships)}")

    def clean_and_validate(self, entities: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
        cleaned_entities = []
        for entity in entities:
            try:
                cleaned_entity = dict(
                    entity_name=clean_str(entity['entity_name'].upper()),
                    entity_type=clean_str(entity['entity_type'].upper()),
                    description=clean_str(entity['description']),
                    importance_score=float(entity['importance_score'])
                )
                cleaned_entities.append(cleaned_entity)
            except ValueError as e:
                logger.warning(f"Invalid entity: {entity}. Error: {str(e)}")

        cleaned_relationships = []
        for relationship in relationships:
            try:
                cleaned_relationship = dict(
                    src_id=clean_str(relationship['src_id'].upper()),
                    tgt_id=clean_str(relationship['tgt_id'].upper()),
                    description=clean_str(relationship['description']),
                    weight=float(relationship['weight']),
                    order=int(relationship['order'])
                )
                cleaned_relationships.append(cleaned_relationship)
            except ValueError as e:
                logger.warning(f"Invalid relationship: {relationship}. Error: {str(e)}")

        direct_relationships = sum(1 for r in cleaned_relationships if r['order'] == 1)
        second_order_relationships = sum(1 for r in cleaned_relationships if r['order'] == 2)
        third_order_relationships = sum(1 for r in cleaned_relationships if r['order'] == 3)
        logger.debug(f"Direct Relationships: {direct_relationships} | Second-order: {second_order_relationships} | Third-order: {third_order_relationships} | Total Relationships: {len(cleaned_relationships)}")

        return cleaned_entities, cleaned_relationships
