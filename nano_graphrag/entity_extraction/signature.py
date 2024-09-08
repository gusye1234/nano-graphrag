import dspy


class EntityTypeExtraction(dspy.Signature):
    """Signature for extracting entity types from input text."""

    input_text = dspy.InputField(desc="The text to extract entity types from.")
    entity_types = dspy.OutputField(
        desc="""
        List of entity types present in the text separated by commas and make sure they are single word, unique, 
        and important based on the text's context.
        For instance: [person, event, technology, mission, organization, location].
        """
    )


class EntityExtraction(dspy.Signature):
    """Signature for extracting entities from input text."""

    input_text = dspy.InputField(desc="The text to extract entities from.")
    entity_types = dspy.InputField(desc="List of entity types to consider.")
    entities = dspy.OutputField(
        desc="""
        List of extracted entities including their types, descriptions, and importance scores (0-1, with 1 being most important). 
        Format should be a list of dictionaries like the following:
        [
            {
                "name": "Entity name",
                "type": "Entity type",
                "description": "Detailed and specific description",
                "importance_score": float (0.0 to 1.0)
            }
        ]
        Make sure descriptions are detailed and specific, and all entity types are included mentioned from the text. 
        Ensure that all fields in the above format are present for every single entity dictionary.
        Entities must have an importance score greater than 0.5, which means both primary and secondary importance entities will be extracted.
        """
    )


class RelationshipExtraction(dspy.Signature):
    """Signature for extracting relationships between entities from input text."""

    input_text = dspy.InputField(desc="The text to extract relationships from.")
    entities = dspy.InputField(
        desc="""
        List of extracted entities including their types, descriptions, and importance scores (0-1, with 1 being most important).
        Format should be a list of dictionaries like the following:
        [
            {
                "name": "Entity name",
                "type": "Entity type",
                "description": "Detailed and specific description",
                "importance_score": float (0.0 to 1.0)
            }
        ]
        """
    )
    relationships = dspy.OutputField(
        desc="""
        List of relationships between entities, including detailed descriptions and importance scores (0-1, with 1 being most important). 
        Format should be a list of dictionaries like the following:
        [
            {
                "source": "Source entity name",
                "target": "Target entity name",
                "description": "Detailed description of the relationship",
                "importance_score": float (0.0 to 1.0)
            }
        ]
        Make sure relationships are detailed and specific.
        Ensure that all fields in the above format are present for every single relationship dictionary.
        """
    )

