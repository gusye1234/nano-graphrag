import dspy
from pydantic import BaseModel, Field


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
        Make sure descriptions are concise and specific, and all entity types are included from the text. 
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
                    "weight": 0.7
                },
                ...
            ]
        }
        Make sure relationships are detailed and specific.
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
                    "weight": 0.7
                },
                ...
            ]
        }
        More specifically:
        1. Direct relationships between entities that were not captured initially.
        2. Implicit relationships that can be inferred from the context.
        3. Secondary or tertiary relationships that provide important connections.
        4. Hierarchical, causal, or temporal relationships that may have been overlooked.
        5. Relationships involving the newly identified missing entities.
        6. First-order consequences: The immediate and direct effects of an action or decision.
        7. Second-order consequences: The indirect effects that result from the first-order consequences.
        8. Third-order consequences: The further indirect effects that result from the second-order consequences.
        Only include relationships between entities in the combined entities list (extracted + missing).
        Ensure the output is strictly JSON formatted without any trailing text or comments.
        """
    )
