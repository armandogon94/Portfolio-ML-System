"""PyTorch LSTM for time series demand forecasting."""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """LSTM-based demand forecasting model.

    Input: (batch, window_size, 1) - past demand values
    Output: (batch, forecast_horizon) - future demand predictions
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 7,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        prediction = self.fc(last_output)  # (batch, forecast_horizon)
        return prediction
