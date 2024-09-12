import dspy
import numpy as np


class AssessRelationship(dspy.Signature):
    """Assess the similarity of two relationships."""
    gold_relationship = dspy.InputField()
    predicted_relationship = dspy.InputField()
    similarity_score = dspy.OutputField(desc="Similarity score between 0 and 1")


def relationship_similarity_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    similarity_scores = []
    
    for gold_rel, pred_rel in zip(gold.relationships.context, pred.relationships.context):
        assessment = dspy.Predict(AssessRelationship)(
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
    true_set = set(item.entity_name for item in gold.entities.context)
    pred_set = set(item.entity_name for item in pred.entities.context)
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall
