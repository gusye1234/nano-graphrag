import dspy
from pydantic import BaseModel, Field
from nano_graphrag._utils import clean_str


"""
Obtained from:
https://github.com/SciPhi-AI/R2R/blob/6e958d1e451c1cb10b6fc868572659785d1091cb/r2r/providers/prompts/defaults.jsonl
"""
ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENTAGE",
    "PRODUCT",
    "EVENT",
    "LANGUAGE",
    "NATIONALITY",
    "RELIGION",
    "TITLE",
    "PROFESSION",
    "ANIMAL",
    "PLANT",
    "DISEASE",
    "MEDICATION",
    "CHEMICAL",
    "MATERIAL",
    "COLOR",
    "SHAPE",
    "MEASUREMENT",
    "WEATHER",
    "NATURAL_DISASTER",
    "AWARD",
    "LAW",
    "CRIME",
    "TECHNOLOGY",
    "SOFTWARE",
    "HARDWARE",
    "VEHICLE",
    "FOOD",
    "DRINK",
    "SPORT",
    "MUSIC_GENRE",
    "INSTRUMENT",
    "ARTWORK",
    "BOOK",
    "MOVIE",
    "TV_SHOW",
    "ACADEMIC_SUBJECT",
    "SCIENTIFIC_THEORY",
    "POLITICAL_PARTY",
    "CURRENCY",
    "STOCK_SYMBOL",
    "FILE_TYPE",
    "PROGRAMMING_LANGUAGE",
    "MEDICAL_PROCEDURE",
    "CELESTIAL_BODY",
]


class Entity(BaseModel):
    entity_name: str = Field(..., description="The name of the entity.")
    entity_type: str = Field(..., description="The type of the entity.")
    description: str = Field(
        ..., description="The description of the entity, in details and comprehensive."
    )
    importance_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Importance score of the entity. Should be between 0 and 1 with 1 being the most important.",
    )


class Relationship(BaseModel):
    src_id: str = Field(..., description="The name of the source entity.")
    tgt_id: str = Field(..., description="The name of the target entity.")
    description: str = Field(
        ...,
        description="The description of the relationship between the source and target entity, in details and comprehensive.",
    )
    weight: float = Field(
        ...,
        ge=0,
        le=1,
        description="The weight of the relationship. Should be between 0 and 1 with 1 being the strongest relationship.",
    )
    order: int = Field(
        ...,
        ge=1,
        le=3,
        description="The order of the relationship. 1 for direct relationships, 2 for second-order, 3 for third-order.",
    )


class CombinedExtraction(dspy.Signature):
    """
    Given a text document that is potentially relevant to this activity and a list of entity types,
    identify all entities of those types from the text and all relationships among the identified entities.

    Entity Guidelines:
    1. Each entity name should be an actual atomic word from the input text.
    2. Avoid duplicates and generic terms.
    3. Make sure descriptions are detailed and comprehensive. Use multiple complete sentences for each point below:
        a). The entity's role or significance in the context
        b). Key attributes or characteristics
        c). Relationships to other entities (if applicable)
        d). Historical or cultural relevance (if applicable)
        e). Any notable actions or events associated with the entity
    4. All entity types from the text must be included.
    5. IMPORTANT: Only use entity types from the provided 'entity_types' list. Do not introduce new entity types.

    Relationship Guidelines:
    1. Make sure relationship descriptions are detailed and comprehensive. Use multiple complete sentences for each point below:
        a). The nature of the relationship (e.g., familial, professional, causal)
        b). The impact or significance of the relationship on both entities
        c). Any historical or contextual information relevant to the relationship
        d). How the relationship evolved over time (if applicable)
        e). Any notable events or actions that resulted from this relationship
    2. Include direct relationships (order 1) as well as higher-order relationships (order 2 and 3):
        a). Direct relationships: Immediate connections between entities.
        b). Second-order relationships: Indirect effects or connections that result from direct relationships.
        c). Third-order relationships: Further indirect effects that result from second-order relationships.
    3. The "src_id" and "tgt_id" fields must exactly match entity names from the extracted entities list.
    """

    input_text: str = dspy.InputField(
        desc="The text to extract entities and relationships from."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of entity types used for extraction."
    )
    entities: list[Entity] = dspy.OutputField(
        desc="List of entities extracted from the text and the entity types."
    )
    relationships: list[Relationship] = dspy.OutputField(
        desc="List of relationships extracted from the text and the entity types."
    )


class CritiqueCombinedExtraction(dspy.Signature):
    input_text: str = dspy.InputField(
        desc="The text to extract entities and relationships from."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of entity types used for extraction."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of entities extracted from the text and the entity types."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of relationships extracted from the text and the entity types."
    )
    entity_critique: str = dspy.OutputField(desc="The critique of the current entities.")
    relationship_critique: str = dspy.OutputField(desc="The critique of the current relationships.")


class RefineCombinedExtraction(dspy.Signature):
    input_text: str = dspy.InputField(
        desc="The text to extract entities and relationships from."
    )
    entity_types: list[str] = dspy.InputField(
        desc="List of entity types used for extraction."
    )
    current_entities: list[Entity] = dspy.InputField(
        desc="List of entities extracted from the text and the entity types."
    )
    current_relationships: list[Relationship] = dspy.InputField(
        desc="List of relationships extracted from the text and the entity types."
    )
    entity_critique: str = dspy.InputField(desc="The critique of the current entities.")
    relationship_critique: str = dspy.InputField(desc="The critique of the current relationships.")
    refined_entities: list[Entity] = dspy.OutputField(
        desc="List of entities extracted from the text and the entity types, improved based on the critique and current entities."
    )
    refined_relationships: list[Relationship] = dspy.OutputField(
        desc="List of relationships extracted from the text and the entity types, improved based on the critique and current relationships."
    )


class TypedEntityRelationshipExtractorException(dspy.Module):
    def __init__(
        self,
        predictor: dspy.Module,
        exception_types: tuple[type[Exception]] = (Exception,),
    ):
        super().__init__()
        self.predictor = predictor
        self.exception_types = exception_types

    def copy(self):
        return TypedEntityRelationshipExtractorException(self.predictor)

    def forward(self, **kwargs):
        try:
            prediction = self.predictor(**kwargs)
            return prediction

        except Exception as e:
            if isinstance(e, self.exception_types):
                return dspy.Prediction(entities=[], relationships=[])

            raise e


class TypedEntityRelationshipExtractor(dspy.Module):
    def __init__(
        self,
        lm: dspy.LM = None,
        max_retries: int = 3,
        entity_types: list[str] = ENTITY_TYPES
    ):
        super().__init__()
        self.lm = lm
        self.entity_types = entity_types
        self.extractor = dspy.TypedChainOfThought(signature=CombinedExtraction, max_retries=max_retries)
        self.extractor = TypedEntityRelationshipExtractorException(
            self.extractor, exception_types=(ValueError,)
        )

    def forward(self, input_text: str) -> dspy.Prediction:
        with dspy.context(lm=self.lm if self.lm is not None else dspy.settings.lm):
            extraction_result = self.extractor(
                input_text=input_text, entity_types=self.entity_types
            )

        entities = [
            dict(
                entity_name=clean_str(entity.entity_name.upper()),
                entity_type=clean_str(entity.entity_type.upper()),
                description=clean_str(entity.description),
                importance_score=float(entity.importance_score),
            )
            for entity in extraction_result.entities
        ]

        relationships = [
            dict(
                src_id=clean_str(relationship.src_id.upper()),
                tgt_id=clean_str(relationship.tgt_id.upper()),
                description=clean_str(relationship.description),
                weight=float(relationship.weight),
                order=int(relationship.order),
            )
            for relationship in extraction_result.relationships
        ]

        return dspy.Prediction(entities=entities, relationships=relationships)


class SelfRefineEntityRelationshipExtractor(dspy.Module):
    def __init__(
            self,         
            lm: dspy.LM = None,
            max_retries: int = 3,
            num_turns: int = 1,
            entity_types: list[str] = ENTITY_TYPES
        ):
        super().__init__()
        self.lm = lm
        self.zero_shot = dspy.TypedChainOfThought(
            signature=CombinedExtraction, 
            max_retries=max_retries
        )
        self.critique = dspy.TypedChainOfThought(
            signature=CritiqueCombinedExtraction, 
            max_retries=max_retries
        )
        self.refine =  dspy.TypedChainOfThought(
            signature=RefineCombinedExtraction, 
            max_retries=max_retries
        )
        self.num_turns = num_turns
        self.entity_types = entity_types

    def forward(self, input_text: str) -> dspy.Prediction:
        with dspy.context(lm=self.lm if self.lm is not None else dspy.settings.lm):
            initial_result = self.zero_shot(input_text=input_text, entity_types=self.entity_types)
            current_entities = initial_result.entities
            current_relationships = initial_result.relationships
            
            for _ in range(self.num_turns):
                critique_result = self.critique(
                    input_text=input_text, 
                    entity_types=self.entity_types, 
                    current_entities=current_entities,
                    current_relationships=current_relationships
                )
                refined_result = self.refine(
                    input_text=input_text, 
                    entity_types=self.entity_types, 
                    current_entities=current_entities,
                    current_relationships=current_relationships,
                    entity_critique=critique_result.entity_critique,
                    relationship_critique=critique_result.relationship_critique
                )
                current_entities = refined_result.refined_entities if refined_result.refined_entities else current_entities
                current_relationships = refined_result.refined_relationships if refined_result.refined_relationships else current_relationships
        
        entities = [
            dict(
                entity_name=clean_str(entity.entity_name.upper()),
                entity_type=clean_str(entity.entity_type.upper()),
                description=clean_str(entity.description),
                importance_score=float(entity.importance_score),
            )
            for entity in current_entities
        ]
        relationships = [
            dict(
                src_id=clean_str(relationship.src_id.upper()),
                tgt_id=clean_str(relationship.tgt_id.upper()),
                description=clean_str(relationship.description),
                weight=float(relationship.weight),
                order=int(relationship.order),
            )
            for relationship in current_relationships
        ]
        return dspy.Prediction(entities=entities, relationships=relationships)
    