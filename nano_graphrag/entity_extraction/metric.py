import dspy
import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


def relationship_similarity_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    true_dict = {(item.src_id, item.tgt_id): item.description for item in gold.relationships.context}
    pred_dict = {(item.src_id, item.tgt_id): item.description for item in pred.relationships.context}
    common_keys = set(true_dict.keys()) & set(pred_dict.keys())

    if not common_keys:
        return 0.0

    true_descs = [true_dict[k] for k in common_keys]
    pred_descs = [pred_dict[k] for k in common_keys]
    true_embeddings = local_embedding(true_descs)
    pred_embeddings = local_embedding(pred_descs)
    similarities = np.dot(true_embeddings, pred_embeddings.T)
    return np.mean(similarities.diagonal())


def relationship_recall_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    true_set = set((item.src_id, item.tgt_id) for item in gold.relationships.context)
    pred_set = set((item.src_id, item.tgt_id) for item in pred.relationships.context)
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def entity_recall_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    true_set = set(item.entity_name for item in gold.entities.context)
    pred_set = set(item.entity_name for item in pred.entities.context)
    true_positives = len(pred_set.intersection(true_set))
    false_negatives = len(true_set - pred_set)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall
