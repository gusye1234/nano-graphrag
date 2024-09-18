import dspy
import numpy as np


class AssessRelationship(dspy.Signature):
    """
    Crucial considerations when assessing the similarity of two relationships:
    - Take the "src_id" and "tgt_id" fields into account as the source and target entities are crucial for assessing the relationship similarity.
    - Take the "description" field into account as it contains detailed information about the relationship.
    """

    gold_relationship = dspy.InputField(
        desc="""
        The gold-standard relationship to compare against.

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
    predicted_relationship = dspy.InputField(
        desc="""
        The predicted relationship to compare against.

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
    similarity_score = dspy.OutputField(
        desc="""
        Similarity score of the predicted relationship to the gold-standard relationship between 0 and 1, 1 being the highest similarity
        """
    )


def relationship_similarity_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    similarity_scores = []
    model = dspy.ChainOfThought(AssessRelationship)

    for gold_rel, pred_rel in zip(gold['relationships'], pred['relationships']):
        assessment = model(
            gold_relationship=gold_rel,
            predicted_relationship=pred_rel
        )
        
        try:
            score = float(assessment.similarity_score)
            similarity_scores.append(score)
        except ValueError:
            similarity_scores.append(0.0)
    
    return np.mean(similarity_scores) if similarity_scores else 0.0


def entity_recall_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    true_set = set(item['entity_name'] for item in gold['entities'])
    pred_set = set(item['entity_name'] for item in pred['entities'])
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall
