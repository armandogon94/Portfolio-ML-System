"""Unsupervised fraud baseline: a PyTorch autoencoder trained on legitimate rows.

This is the one genuinely non-generic trainer in the repo, which is why it did not
collapse into ``tabular.py``. It is also the **only** model here that uses MPS.

What it is for: a *baseline*, not the headline. Trained on legitimate transactions
only, it scores anomalies by reconstruction error. The value it adds to the results
table is showing how much a supervised model gains over pure anomaly detection —
which on IEEE-CIS is a lot, and saying so is the point.

Hardware facts that shaped this file (measured on the target machine — Apple
Silicon, 4 performance + 6 efficiency cores, 32 GB, torch 2.13.0):

* MPS gives roughly 1.9-2.2x over CPU on dense matmul. Not 5-10x. A 12-layer MLP
  on 500k x 100 floats is minutes either way.
* ``torch.get_num_threads()`` defaults to 4, not 10.
* **torch 2.13.0 MPS bug, reproduced twice:** ``torch.nn.MultiheadAttention`` hangs
  on MPS, and a CPU tensor loop deadlocked at 0% CPU after a preceding MPS matmul
  in the *same* process. This module contains no attention layer so it does not
  trip the bug — but it never mixes devices in one process either, and any future
  sequence model here must run in its own process per device.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import make_splits
from src.evaluation.classification_metrics import compute_classification_metrics
from src.models.autoencoder import FraudAutoencoder
from src.training.tabular import TabularTrainer, _git_sha

logger = logging.getLogger(__name__)


class AutoencoderTrainer(TabularTrainer):
    """Train the unsupervised fraud baseline.

    Reuses :class:`~src.training.tabular.TabularTrainer` for data loading, feature
    engineering and the split, then substitutes an unsupervised fit/score loop.

    Args:
        config_name: Normally ``"fraud"``.
        use_wandb: See :class:`~src.training.trainer.BaseTrainer`.
        sample: Train on the CI fixture; suppresses checkpoint writes.
        epochs: Training epochs. The default is small on purpose — this is a
            baseline and the owner's machine is not a training cluster.
    """

    def __init__(
        self,
        config_name: str = "fraud",
        *,
        use_wandb: bool = False,
        sample: bool = False,
        epochs: int = 20,
    ):
        super().__init__(config_name, use_wandb=use_wandb, sample=sample)
        self.epochs = epochs
        self.scaler: Any = None
        self.threshold: float = float("nan")

    def train(self, data: dict[str, Any]) -> FraudAutoencoder:
        """Fit on legitimate rows only, then set the anomaly threshold.

        Training on legitimate rows is the whole idea: the network learns what
        normal looks like and fraud shows up as reconstruction error. Including
        fraud in training teaches it to reconstruct fraud too.
        """
        import torch
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from src.device import get_device

        device = get_device()
        logger.info("Autoencoder device: %s", device)

        numeric = data["X_train"].select_dtypes(include=[np.number])
        self.numeric_columns = list(numeric.columns)
        self.scaler = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )

        legitimate = numeric[data["y_train"] == 0]
        matrix = self.scaler.fit_transform(legitimate).astype(np.float32)

        model = FraudAutoencoder(
            input_dim=matrix.shape[1],
            hidden_dims=self.config["model"]["params"].get("hidden_dims", [64, 32, 16]),
        ).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        tensor = torch.from_numpy(matrix).to(device)
        batch_size = int(self.config["model"]["params"].get("batch_size", 512))

        model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(tensor.shape[0], device=device)
            total = 0.0
            for start in range(0, tensor.shape[0], batch_size):
                batch = tensor[permutation[start : start + batch_size]]
                optimiser.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimiser.step()
                total += float(loss.item()) * batch.shape[0]
            logger.info(
                "epoch %2d/%d  train_mse=%.6f",
                epoch + 1,
                self.epochs,
                total / tensor.shape[0],
            )

        model.eval()
        # Threshold at the configured percentile of TRAIN reconstruction error.
        # Setting it from test error would be leakage.
        percentile = float(self.config["model"]["params"].get("threshold_percentile", 95))
        with torch.no_grad():
            train_errors = model.reconstruction_error(tensor).cpu().numpy()
        self.threshold = float(np.percentile(train_errors, percentile))
        logger.info("Anomaly threshold (p%.0f of train error) = %.6f", percentile, self.threshold)
        return model

    def evaluate(self, model: FraudAutoencoder, data: dict[str, Any]) -> dict[str, float]:
        """Score the test split by reconstruction error, ranked as a probability."""
        import torch

        from src.device import get_device

        device = get_device()
        numeric = data["X_test"].reindex(columns=self.numeric_columns)
        matrix = self.scaler.transform(numeric).astype(np.float32)
        with torch.no_grad():
            errors = model.reconstruction_error(torch.from_numpy(matrix).to(device)).cpu().numpy()

        # Reconstruction error is unbounded; the metrics want something in [0, 1].
        # Rank-normalising preserves the ordering exactly, so PR-AUC and ROC-AUC
        # are unaffected, while Brier score becomes interpretable.
        scores = pd.Series(errors).rank(pct=True).to_numpy()

        metrics = compute_classification_metrics(data["y_test"], scores, prefix="test")
        metrics["anomaly_threshold"] = self.threshold
        metrics["model_family"] = 0.0  # marker: unsupervised, see reports/RESULTS.md
        return metrics

    def run(self) -> dict[str, float]:
        """Train, evaluate and checkpoint the unsupervised baseline."""
        started = time.time()
        logger.info("Autoencoder baseline for %s (sample=%s)", self.problem, self.sample)

        frame = self.load_data()
        split = make_splits(frame, self.config["split"], self.config["data"]["target"], self.seed)[
            0
        ]
        data = self.build_matrix(frame, split)
        model = self.train(data)
        metrics = self.evaluate(model, data)
        metrics["training_time_seconds"] = round(time.time() - started, 1)

        self.metrics = metrics
        self.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        self._save(model, metrics)
        self.finish()
        return metrics

    def _save(self, model: FraudAutoencoder, metrics: dict[str, float]) -> Path | None:
        """Write the autoencoder checkpoint under ``checkpoints/<problem>_autoencoder/``."""
        if self.sample:
            logger.warning("SAMPLE MODE: skipping autoencoder checkpoint.")
            return None

        import joblib
        import torch

        directory = Path(self.config["training"]["checkpoint_dir"] + "_autoencoder")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), directory / "autoencoder.pt")
        joblib.dump(
            {"scaler": self.scaler, "numeric_columns": self.numeric_columns},
            directory / "features.joblib",
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "problem": f"{self.problem}_autoencoder",
                    "model_type": "autoencoder",
                    "family": "unsupervised",
                    "seed": self.seed,
                    "git_sha": _git_sha(),
                    "epochs": self.epochs,
                    "metrics": metrics,
                    "note": (
                        "Unsupervised baseline trained on legitimate rows only. "
                        "Reported to show what the supervised model adds, not as "
                        "the headline result."
                    ),
                },
                indent=2,
                default=str,
            )
        )
        logger.info("Wrote %s", directory)
        return directory


def load_feature_module(config: dict) -> Any:
    """Import the feature module named in a config. Shared by serving."""
    return importlib.import_module(config["features"]["module"])
