import dspy
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


class Entity(BaseModel):
    entity_name: str = Field(..., description="Cleaned and uppercased entity name, strictly upper case")
    entity_type: str = Field(..., description="Cleaned and uppercased entity type, strictly upper case")
    description: str = Field(..., description="Detailed and specific description of the entity")
    importance_score: float = Field(ge=0.0, le=1.0, description="0 to 1, with 1 being most important")


class Relationship(BaseModel):
    src_id: str = Field(..., description="Cleaned and uppercased source entity, strictly upper case")
    tgt_id: str = Field(..., description="Cleaned and uppercased target entity, strictly upper case")
    description: str = Field(..., description="Detailed and specific description of the relationship")
    weight: float = Field(ge=0.0, le=1.0, description="0 to 1, with 1 being most important")
    order: int = Field(..., description="1 for direct relationships, 2 for second-order, 3 for third-order, etc")


class Entities(BaseModel):
    context: list[Entity]


class Relationships(BaseModel):
    context: list[Relationship]


class CombinedExtraction(dspy.Signature):
    """Signature for extracting both entities and relationships from input text."""

    input_text: str = dspy.InputField(desc="The text to extract entities and relationships from.")
    entity_types: EntityTypes = dspy.InputField()
    entities: Entities = dspy.OutputField(
        desc="""
        Format:
        {
            "context": [
                {
                    "entity_name": "ENTITY NAME",
                    "entity_type": "ENTITY TYPE",
                    "description": "Detailed description",
                    "importance_score": 0.8
                },
                ...
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
    relationships: Relationships = dspy.OutputField(
        desc="""
        Format:
        {
            "context": [
                {
                    "src_id": "SOURCE ENTITY",
                    "tgt_id": "TARGET ENTITY",
                    "description": "Detailed description of the relationship",
                    "weight": 0.7,
                    "order": 1  # 1 for direct relationships, 2 for second-order, 3 for third-order, etc.
                },
                ...
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

    input_text: str = dspy.InputField(desc="The original input text.")
    entity_types: EntityTypes = dspy.InputField()
    entities: Entities = dspy.InputField(desc="List of extracted entities.")
    relationships: Relationships = dspy.InputField(desc="List of extracted relationships.")
    missing_entities: Entities = dspy.OutputField(
        desc="""
        Format:
        {
            "context": [
                {
                    "entity_name": "ENTITY NAME",
                    "entity_type": "ENTITY TYPE",
                    "description": "Detailed description",
                    "importance_score": 0.8
                },
                ...
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
    missing_relationships: Relationships = dspy.OutputField(
        desc="""
        Format:
        {
            "context": [
                {
                    "src_id": "SOURCE ENTITY",
                    "tgt_id": "TARGET ENTITY",
                    "description": "Detailed description of the relationship",
                    "weight": 0.7,
                    "order": 1  # 1 for direct, 2 for second-order, 3 for third-order
                },
                ...
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
        all_entities = Entities(context=entities.context + missing_entities.context)
        all_relationships = Relationships(context=relationships.context + missing_relationships.context)
        logger.debug(f"Entities: {len(entities.context)} | Missed Entities: {len(missing_entities.context)} | Total Entities: {len(all_entities.context)}")
        logger.debug(f"Relationships: {len(relationships.context)} | Missed Relationships: {len(missing_relationships.context)} | Total Relationships: {len(all_relationships.context)}")
        
        for entity in all_entities.context:
            entity.entity_name = clean_str(entity.entity_name.upper())
            entity.entity_type = clean_str(entity.entity_type.upper())
            entity.description = clean_str(entity.description)
            entity.importance_score = float(entity.importance_score)

        for relationship in all_relationships.context:
            relationship.src_id = clean_str(relationship.src_id.upper())
            relationship.tgt_id = clean_str(relationship.tgt_id.upper())
            relationship.description = clean_str(relationship.description)
            relationship.weight = float(relationship.weight)
            relationship.order = int(relationship.order)

        direct_relationships = sum(1 for r in all_relationships.context if r.order == 1)
        second_order_relationships = sum(1 for r in all_relationships.context if r.order == 2)
        third_order_relationships = sum(1 for r in all_relationships.context if r.order == 3)
        logger.debug(f"Direct Relationships: {direct_relationships} | Second-order: {second_order_relationships} | Third-order: {third_order_relationships} | Total Relationships: {len(all_relationships.context)}")
        return dspy.Prediction(entities=all_entities, relationships=all_relationships)
    