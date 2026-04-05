"""Tests for model architectures."""

import numpy as np
import torch

from src.models.fraud_autoencoder import FraudAutoencoder
from src.models.lstm_forecaster import LSTMForecaster


def test_autoencoder_shape():
    model = FraudAutoencoder(input_dim=12, hidden_dims=[64, 32, 16])
    x = torch.randn(5, 12)
    output = model(x)
    assert output.shape == (5, 12)


def test_autoencoder_reconstruction_error():
    model = FraudAutoencoder(input_dim=10)
    x = torch.randn(3, 10)
    errors = model.reconstruction_error(x)
    assert errors.shape == (3,)
    assert (errors >= 0).all()


def test_lstm_shape():
    model = LSTMForecaster(input_size=1, hidden_size=32, num_layers=1, forecast_horizon=7)
    x = torch.randn(4, 30, 1)  # batch=4, window=30, features=1
    output = model(x)
    assert output.shape == (4, 7)
