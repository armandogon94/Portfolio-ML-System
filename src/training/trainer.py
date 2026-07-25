"""``BaseTrainer`` — config loading, experiment tracking, and the MLflow registry.

Kept deliberately small. It owns what every trainer needs and knows nothing about
tabular data, splits or metrics: those belong to ``tabular.py`` and
``autoencoder_pipeline.py``, the two concrete trainers.

MLflow is primary for real runs (local file store by default, HTTP when
``MLFLOW_TRACKING_URI`` points at the container). Sample runs are deliberately
untracked. W&B is optional and off unless a real API key is present — see
``docs/adr/0002-experiment-tracking.md``.
"""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import mlflow

from src.config import config_path, load_config
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
        sample: Marks a fixture-only run. Sample mode opens no tracking run, so
            fixture metrics cannot enter any experiment or external tracker.
    """

    def __init__(
        self,
        config_name: str,
        *,
        use_wandb: bool = False,
        sample: bool = False,
        tracking_model_type: str | None = None,
        registry_name: str | None = None,
    ):
        setup_logging()
        # These must exist before either tracker is initialised. In particular,
        # sample mode deliberately opens no MLflow or W&B run.
        self.sample = sample
        self.config_name = config_path(config_name).stem
        self.config = load_config(config_name)
        self.problem: str = self.config["problem"]
        self.seed: int = int(self.config["seed"])
        self.tracking_model_type = tracking_model_type or self.config["model"]["type"]
        self.registry_name = registry_name or self.config_name
        self.metrics: dict[str, float] = {}
        self.mlflow_run_id: str | None = None

        self._seed_everything()
        self.use_wandb = False if self.sample else use_wandb and self._init_wandb()
        self.use_mlflow = False if self.sample else self._init_mlflow()
        if self.sample:
            logger.info("SAMPLE MODE: no MLflow or W&B run was opened.")

    # ── determinism ──────────────────────────────────────────────────────────

    def _seed_everything(self) -> None:
        """Thread the single config seed through every RNG that can affect a run.

        ``PYTHONHASHSEED`` is set for completeness but only takes effect in a
        fresh interpreter. PyTorch weight initialisation, dropout and shuffling
        use their own generator and must be seeded explicitly — but see
        :meth:`seed_torch`, which is why that does not happen here.

        This makes repeated initialisation reproducible; it does not promise
        bit-identical training on every backend. Some MPS kernels remain
        nondeterministic even when their random generator is seeded.
        """
        import random

        import numpy as np

        os.environ.setdefault("PYTHONHASHSEED", str(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.seed_torch()

    def seed_torch(self) -> bool:
        """Seed PyTorch's RNGs, but **only if torch is already imported**.

        Importing torch here would be a one-line change that breaks every
        gradient-boosted run on this machine. PyTorch ships its own ``libomp``;
        LightGBM links another. Loading both into one process on macOS arm64
        segfaults inside ``LGBMClassifier.fit``. Measured on 2026-07-25 against
        the real 8,101-row churn training matrix, torch 2.x and LightGBM 4.6.0:

            python -c "import pandas as pd, lightgbm as lgb; ...; m.fit(X, y)"
                -> OK
            python -c "import torch; import pandas as pd, lightgbm as lgb; ...; m.fit(X, y)"
                -> Segmentation fault: 11

        So the tabular path must never pull torch in, and the autoencoder path —
        which imports torch for its own reasons and never touches LightGBM —
        calls this after that import and gets a properly seeded RNG.

        Returns:
            True when torch was present and seeded, False when it was skipped.
        """
        torch = sys.modules.get("torch")
        if torch is None:
            logger.debug("torch is not imported; skipping its seeding (see seed_torch docs).")
            return False

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        mps = getattr(torch, "mps", None)
        mps_manual_seed = getattr(mps, "manual_seed", None)
        if callable(mps_manual_seed):
            mps_manual_seed(self.seed)
        return True

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
                tags=[self.problem, self.tracking_model_type],
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
            mlflow.set_tags(
                {
                    "problem": self.problem,
                    "sample": str(self.sample).lower(),
                    "config": self.config_name,
                    "model": self.tracking_model_type,
                }
            )
            mlflow.log_params(_flatten(self.config))
            logger.info("MLflow run %s at %s", self.mlflow_run_id, tracking_uri)
            return True
        except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
            logger.warning("MLflow init failed (%s). Continuing without tracking.", exc)
            try:
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("MLflow cleanup after init failure also failed: %s", cleanup_exc)
            return False

    def _run_name(self) -> str:
        return f"{self.config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record metrics locally; tracker failures are loud but non-fatal."""
        self.metrics.update(metrics)
        clean = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v  # drop NaN: MLflow rejects it
        }
        if self.use_mlflow and clean:
            try:
                mlflow.log_metrics(clean)
            except Exception as exc:  # noqa: BLE001 - tracking must never lose a checkpoint
                logger.warning(
                    "MLflow metric logging failed after artifacts were saved: %s. "
                    "The local checkpoint remains valid.",
                    exc,
                )
        if self.use_wandb and clean:
            try:
                import wandb

                wandb.log(clean)
            except Exception as exc:  # noqa: BLE001 - tracking must never lose a checkpoint
                logger.warning(
                    "W&B metric logging failed after artifacts were saved: %s. "
                    "The local checkpoint remains valid.",
                    exc,
                )

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
            mlflow.log_artifacts(artifact_dir, artifact_path="checkpoint")
            client = mlflow.tracking.MlflowClient()
            try:
                client.create_registered_model(self.registry_name)
            except mlflow.exceptions.MlflowException:
                pass  # already registered
            version = client.create_model_version(
                name=self.registry_name,
                source=f"runs:/{self.mlflow_run_id}/checkpoint",
                run_id=self.mlflow_run_id,
            )
            logger.info("Registered %s v%s in MLflow", self.registry_name, version.version)
            return str(version.version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model registration failed: %s", exc)
            return None

    def finish(self) -> None:
        """Close out active trackers without masking a completed training run."""
        if self.use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception as exc:  # noqa: BLE001
                logger.warning("W&B finish failed: %s", exc)
        if self.use_mlflow and mlflow.active_run():
            try:
                mlflow.end_run()
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow end_run failed: %s", exc)

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
