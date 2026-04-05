"""PyTorch Autoencoder for fraud detection via reconstruction error."""

import torch
import torch.nn as nn


class FraudAutoencoder(nn.Module):
    """Symmetric autoencoder: learns to reconstruct normal transactions.

    Fraudulent transactions produce higher reconstruction error.
    Architecture: Input -> 64 -> 32 -> 16 -> 32 -> 64 -> Output
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        for dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error."""
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)
