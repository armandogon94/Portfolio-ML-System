"""Classification evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    prefix: str = "test",
) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (binary).
        y_prob: Predicted probabilities (for AUC).
        prefix: Metric name prefix (e.g., 'test', 'train').
    """
    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics[f"{prefix}_auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass

    return metrics
