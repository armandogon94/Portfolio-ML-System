"""``BaseTrainer`` — config loading, experiment tracking, and the MLflow registry.

Kept deliberately small. It owns what every trainer needs and knows nothing about
tabular data, splits or metrics: those belong to ``tabular.py`` and
``autoencoder_pipeline.py``, the two concrete trainers.

MLflow is primary and always-on (local file store by default, HTTP when
``MLFLOW_TRACKING_URI`` points at the container). W&B is optional and off unless a
real API key is present — see ``docs/adr/0002-experiment-tracking.md``.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import mlflow

from src.config import load_config
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

#: Placeholder shipped in ``.env.example``. Treated as "unset" so a user who
#: copies the example file does not get confusing W&B auth failures.
_WANDB_PLACEHOLDER = "your_key_here"


class BaseTrainer(ABC):
    """Shared scaffolding for a training run.

    Args:
        config_name: Problem name resolving to ``configs/<name>.yaml``.
        use_wandb: Attempt W&B logging. Ignored when no real API key is set.
    """

    def __init__(self, config_name: str, *, use_wandb: bool = False):
        setup_logging()
        self.config = load_config(config_name)
        self.problem: str = self.config["problem"]
        self.seed: int = int(self.config["seed"])
        self.metrics: dict[str, float] = {}
        self.mlflow_run_id: str | None = None

        self._seed_everything()
        self.use_wandb = use_wandb and self._init_wandb()
        self.use_mlflow = self._init_mlflow()

    # ── determinism ──────────────────────────────────────────────────────────

    def _seed_everything(self) -> None:
        """Thread the single config seed through every RNG that can affect a run.

        ``PYTHONHASHSEED`` is set for completeness but only takes effect in a
        fresh interpreter; the value that actually matters here is numpy's, which
        every estimator in the registry derives from.
        """
        import random

        import numpy as np

        os.environ.setdefault("PYTHONHASHSEED", str(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)

    # ── tracking ─────────────────────────────────────────────────────────────

    def _init_wandb(self) -> bool:
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key or api_key == _WANDB_PLACEHOLDER:
            logger.info("No W&B API key — MLflow only.")
            return False
        try:
            import wandb

            wandb.init(
                project=self.config.get("training", {}).get(
                    "mlflow_experiment", "fintech-ml-system"
                ),
                name=self._run_name(),
                config=self.config,
                tags=[self.problem, self.config["model"]["type"]],
            )
            return True
        except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
            logger.warning("W&B init failed (%s). Continuing with MLflow only.", exc)
            return False

    def _init_mlflow(self) -> bool:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(
                self.config.get("training", {}).get("mlflow_experiment", "fintech-ml-system")
            )
            run = mlflow.start_run(run_name=self._run_name())
            self.mlflow_run_id = run.info.run_id
            mlflow.set_tags({"problem": self.problem, "model": self.config["model"]["type"]})
            mlflow.log_params(_flatten(self.config))
            logger.info("MLflow run %s at %s", self.mlflow_run_id, tracking_uri)
            return True
        except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
            logger.warning("MLflow init failed (%s). Continuing without tracking.", exc)
            return False

    def _run_name(self) -> str:
        return f"{self.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record metrics locally and in whichever trackers are active."""
        self.metrics.update(metrics)
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v  # drop NaN: MLflow rejects it
        }
        if self.use_mlflow and clean:
            mlflow.log_metrics(clean)
        if self.use_wandb and clean:
            import wandb

            wandb.log(clean)

    def register_model(self, artifact_dir: str) -> str | None:
        """Log the checkpoint to MLflow and create a registry model version.

        Returns:
            The new model version, or ``None`` when MLflow is unavailable. A
            tracking failure is logged and swallowed: losing a registry entry is
            not a reason to lose a trained model.
        """
        if not (self.use_mlflow and mlflow.active_run()):
            return None
        try:
            mlflow.log_artifacts(artifact_dir)
            client = mlflow.tracking.MlflowClient()
            try:
                client.create_registered_model(self.problem)
            except mlflow.exceptions.MlflowException:
                pass  # already registered
            version = client.create_model_version(
                name=self.problem,
                source=f"runs:/{self.mlflow_run_id}",
                run_id=self.mlflow_run_id,
            )
            logger.info("Registered %s v%s in MLflow", self.problem, version.version)
            return str(version.version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model registration failed: %s", exc)
            return None

    def finish(self) -> None:
        """Close out the active tracking runs."""
        if self.use_wandb:
            import wandb

            wandb.finish()
        if self.use_mlflow and mlflow.active_run():
            mlflow.end_run()

    # ── contract ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load_data(self) -> Any:
        """Return the canonical frame for this problem."""

    @abstractmethod
    def train(self, data: dict[str, Any]) -> Any:
        """Fit and return the model."""

    @abstractmethod
    def evaluate(self, model: Any, data: dict[str, Any]) -> dict[str, float]:
        """Return the metrics dict for the test split."""

    @abstractmethod
    def run(self) -> dict[str, float]:
        """Execute the full pipeline and return the measured metrics."""


def _flatten(config: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into MLflow-loggable string params."""
    out: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            # MLflow caps param values at 500 chars; a long denylist would 400.
            out[name] = str(value)[:250]
    return out
