from typing import List, Dict
import numpy as np

from sklearn.metrics import (
    precision_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def evaluate_rankings(
    y_true: np.ndarray, y_pred_scores: np.ndarray, k_values: List[int] = [5]
) -> Dict:
    """
        Evaluate rankings using multiple sklearn metrics

    Args:
            y_true: Binary relevance labels (0 or 1)
            y_pred_scores: Predicted scores/probabilities
            k_values: List of k values for P@k and R@k

        Returns:
            Dictionary containing all metrics
    """
    # Sort predictions and get indices
    sorted_indices = np.argsort(y_pred_scores)[::-1]  # Descending order
    y_true_sorted = y_true[sorted_indices]

    metrics = {}

    # Calculate P@k and R@k for different k values
    for k in k_values:
        # Precision at k
        p_at_k = precision_score(
            y_true_sorted[:k],
            np.ones(k),  # All top-k are predicted as positive
            zero_division=0,
        )
        metrics[f"P@{k}"] = p_at_k

        # Recall at k
        r_at_k = np.sum(y_true_sorted[:k]) / np.sum(y_true)
        metrics[f"R@{k}"] = r_at_k

    # Calculate AUC-ROC
    metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_scores)

    # Calculate AUC-PR (Area under Precision-Recall curve)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    metrics["AUC-PR"] = auc(recall, precision)

    # Average Precision
    metrics["AP"] = average_precision_score(y_true, y_pred_scores)
    return metrics
