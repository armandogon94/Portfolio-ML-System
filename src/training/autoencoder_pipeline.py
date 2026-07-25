"""Unsupervised fraud baseline: a PyTorch autoencoder trained on legitimate rows.

This is the one genuinely non-generic trainer in the repo, which is why it did not
collapse into ``tabular.py``. It is also the **only** model here that uses MPS.

What it is for: a *baseline*, not the headline. Trained on legitimate transactions
only, it scores anomalies by reconstruction error. The value it adds to the results
table is showing how much a supervised model gains over pure anomaly detection —
which on IEEE-CIS is a lot, and saying so is the point.

Hardware facts that shaped this file (measured on the target machine — Apple
Silicon, 4 performance + 6 efficiency cores, 32 GB, torch 2.13.0):

* MPS is available for the six-linear-transform autoencoder, but this repository
  has no committed benchmark comparing it with CPU.
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
import platform
import time
from datetime import datetime, timezone
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
        super().__init__(
            config_name,
            use_wandb=use_wandb,
            sample=sample,
            tracking_model_type="autoencoder",
            registry_name=f"{config_name}_autoencoder",
        )
        self.epochs = epochs
        self.scaler: Any = None
        self.threshold: float = float("nan")
        self.estimator_training_rows: int | None = None
        self.model: FraudAutoencoder | None = None

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

        # BaseTrainer.__init__ ran before torch existed in this process, so its
        # torch RNGs are still unseeded. Seed them now, before any weight is
        # initialised. See BaseTrainer.seed_torch for why the import is here and
        # not in the base class.
        self.seed_torch()

        device = get_device()
        logger.info("Autoencoder device: %s", device)

        numeric = data["X_train"].select_dtypes(include=[np.number])
        self.numeric_columns = list(numeric.columns)
        self.scaler = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )

        legitimate = numeric[data["y_train"] == 0]
        matrix = self.scaler.fit_transform(legitimate).astype(np.float32)
        self.estimator_training_rows = len(matrix)

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
        """Score with raw reconstruction error and the train-fitted threshold."""
        import torch

        from src.device import get_device

        device = get_device()
        numeric = data["X_test"].reindex(columns=self.numeric_columns)
        matrix = self.scaler.transform(numeric).astype(np.float32)
        with torch.no_grad():
            errors = model.reconstruction_error(torch.from_numpy(matrix).to(device)).cpu().numpy()

        # Raw error is an honest ranking score for ROC-AUC, PR-AUC and the
        # operational rank metrics. It is not a probability, so calibration
        # metrics such as Brier are intentionally omitted. Hard classifications
        # use the threshold fitted on training reconstruction error.
        metrics = compute_classification_metrics(
            data["y_test"],
            errors,
            threshold=self.threshold,
            prefix="test",
            calibrated=False,
        )
        metrics["anomaly_threshold"] = self.threshold
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
        self.model = model
        metrics = self.evaluate(model, data)
        metrics["training_time_seconds"] = round(time.time() - started, 1)

        self.metrics = metrics
        checkpoint_dir = self._save(model, metrics)
        self.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if checkpoint_dir is not None:
            self.register_model(str(checkpoint_dir))
        self.finish()
        return metrics

    def _save(self, model: FraudAutoencoder, metrics: dict[str, float]) -> Path | None:
        """Write the autoencoder checkpoint under ``checkpoints/<problem>_autoencoder/``."""
        if self.sample:
            logger.warning("SAMPLE MODE: skipping autoencoder checkpoint.")
            return None

        import joblib
        import torch

        base_directory = Path(self.config["training"]["checkpoint_dir"])
        directory = base_directory.with_name(f"{base_directory.name}_autoencoder")
        directory.mkdir(parents=True, exist_ok=True)
        hidden_dims = self.config["model"]["params"].get("hidden_dims", [64, 32, 16])
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": len(self.numeric_columns),
                "hidden_dims": hidden_dims,
                "dropout": 0.1,
            },
            directory / "model.pt",
        )
        joblib.dump(
            {
                "feature_columns": self.numeric_columns,
                "artifacts": self.feature_artifacts,
                "category_dtypes": {},
                "preprocessor": self.scaler,
            },
            directory / "features.joblib",
        )

        metadata = {
            "problem": f"{self.problem}_autoencoder",
            "base_problem": self.problem,
            "config_name": self.config_name,
            "model_type": "autoencoder",
            "family": "unsupervised",
            "seed": self.seed,
            "git_sha": _git_sha(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.config["data"]["source"],
            "split": self.config["split"],
            "feature_columns": self.numeric_columns,
            "n_features": len(self.numeric_columns),
            "metrics": metrics,
            "hyperparameters": {
                "epochs": self.epochs,
                "hidden_dims": hidden_dims,
                "threshold_percentile": self.config["model"]["params"].get(
                    "threshold_percentile", 95
                ),
            },
            "config_file": f"configs/{self.config_name}.yaml",
            "mlflow_run_id": self.mlflow_run_id,
            "checkpoint_fit": {
                "scope": "legitimate_rows_in_training_partition",
                "n_rows": self.estimator_training_rows,
                "evaluation": "held_out_test_partition",
            },
            "evaluation_predictions": {
                "kind": "held_out_test_partition",
                "score": "raw_reconstruction_error",
                "hard_decision_threshold": "95th_percentile_training_reconstruction_error",
            },
            "leakage_controls": {
                "denylist_enforced": True,
                "split_column_excluded": True,
                "threshold_fitted_on_training_data": True,
            },
            "sanity_band_warning": None,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "note": (
                    "The autoencoder may use MPS for training. MPS kernels are "
                    "not guaranteed bit-deterministic even with seeded generators."
                ),
            },
            "note": (
                "Unsupervised baseline trained on legitimate rows only. Raw "
                "reconstruction error supports ranking metrics but is not a "
                "calibrated probability, so no Brier score is reported."
            ),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        reports_dir = Path(self.config["training"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / f"{self.config_name}_autoencoder_metrics.csv"
        pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
            csv_path, index=False
        )
        logger.info("Wrote %s and %s", directory, csv_path)
        return directory


def load_feature_module(config: dict) -> Any:
    """Import the feature module named in a config. Shared by serving."""
    return importlib.import_module(config["features"]["module"])
