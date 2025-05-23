"""Metrics for SpliceAI model evaluation."""

from sklearn.metrics import average_precision_score
import numpy as np


def topk_accuracy(probs, targets_onehot):
    """
    Compute top-k accuracy for each class.
    Args:
        probs (np.ndarray): Predicted probabilities of shape (N, C).
        targets_onehot (np.ndarray): One-hot encoded targets of shape (N, C).
    Returns:
        list: List of top-k accuracies for each class.
    """
    topk_accs = []
    for c in [0, 1, 2]:
        probs_c = probs[:, c]
        targets_c = (targets_onehot[:, c]).astype(int)
        k = targets_c.sum()
        if k == 0:
            return float("nan")
        topk_indices = probs_c.argsort()[::-1][:k]
        correct = targets_c[topk_indices].sum()
        topk_acc = correct / k
        topk_accs.append(topk_acc)
    return topk_accs


def pr_auc(probs, targets_onehot):
    """
    Compute PR-AUC score for each class.
    Args:
        probs (np.ndarray): Predicted probabilities of shape (N, C).
        targets_onehot (np.ndarray): One-hot encoded targets of shape (N, C).
    Returns:
        list: List of average precision scores for each class.
    """
    aucs = []
    for c in [0, 1, 2]:
        auc = average_precision_score(targets_onehot[:, c], probs[:, c])
        aucs.append(auc)
    return aucs


def to_serializable(obj):
    """
    Convert an object to a serializable format.
    Args:
        obj: The object to convert.
    Returns:
        The serializable object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj
