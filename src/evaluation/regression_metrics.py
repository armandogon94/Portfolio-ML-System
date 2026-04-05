"""Regression evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "test",
) -> dict:
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": np.sqrt(mse),
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_r2": r2_score(y_true, y_pred),
        f"{prefix}_mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100),
    }
