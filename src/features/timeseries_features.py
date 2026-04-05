"""Feature engineering for time series demand forecasting."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_sequences(
    data: np.ndarray,
    window_size: int = 30,
    forecast_horizon: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM.

    Args:
        data: 1D array of demand values.
        window_size: Number of past days to use as input.
        forecast_horizon: Number of future days to predict.

    Returns:
        X: shape (n_samples, window_size, 1)
        y: shape (n_samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size : i + window_size + forecast_horizon])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y


class DemandDataset(Dataset):
    """PyTorch Dataset for demand forecasting sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_timeseries(
    df: pd.DataFrame,
    product: str,
    window_size: int = 30,
    forecast_horizon: int = 7,
    test_days: int = 90,
) -> dict:
    """Prepare time series data for a single product.

    Returns dict with train/test splits, scaler, and metadata.
    """
    from sklearn.preprocessing import MinMaxScaler

    product_df = df[df["product_category"] == product].sort_values("date")
    values = product_df["demand"].values.astype(float).reshape(-1, 1)

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    # Split: last test_days for testing
    train_data = scaled[:-test_days]
    test_data = scaled[-test_days - window_size:]  # Include window for first test sample

    X_train, y_train = create_sequences(train_data, window_size, forecast_horizon)
    X_test, y_test = create_sequences(test_data, window_size, forecast_horizon)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "dates": product_df["date"].values,
        "product": product,
    }
