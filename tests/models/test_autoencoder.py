"""The MPS autoencoder: shapes, reconstruction error, and device discipline."""

from __future__ import annotations

import torch

from src.models.autoencoder import FraudAutoencoder


def test_forward_preserves_shape():
    model = FraudAutoencoder(input_dim=12, hidden_dims=[8, 4])
    batch = torch.randn(7, 12)
    assert model(batch).shape == (7, 12)


def test_reconstruction_error_is_per_sample_and_non_negative():
    model = FraudAutoencoder(input_dim=6, hidden_dims=[4, 2])
    batch = torch.randn(5, 6)
    errors = model.reconstruction_error(batch)
    assert errors.shape == (5,)
    assert bool((errors >= 0).all())


def test_error_shrinks_on_data_the_model_was_fitted_to():
    """A trained autoencoder must reconstruct its training distribution better.

    Deliberately CPU-only. torch 2.13.0 on this hardware deadlocked a CPU tensor
    loop that followed an MPS matmul in the same process, so tests never touch MPS.
    """
    torch.manual_seed(42)
    model = FraudAutoencoder(input_dim=4, hidden_dims=[8, 4])
    data = torch.randn(256, 4) * 0.1 + 1.0

    with torch.no_grad():
        before = float(model.reconstruction_error(data).mean())

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(60):
        optimiser.zero_grad()
        loss_fn(model(data), data).backward()
        optimiser.step()
    model.eval()

    with torch.no_grad():
        after = float(model.reconstruction_error(data).mean())
    assert after < before, f"training did not reduce reconstruction error: {before} -> {after}"
