"""Time series evaluation metrics."""

import numpy as np


def compute_timeseries_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "test",
) -> dict:
    """Compute time series forecasting metrics."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)

    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mse": mse,
        f"{prefix}_mape": mape,
    }
