"""Base trainer with W&B + MLflow integration and checkpointing."""

import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console

from src.config import get_project_root, load_config
from src.logging_config import setup_logging

console = Console()
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base class for all model trainers.

    Handles: config loading, W&B + MLflow initialization, checkpoint saving, results CSV export.
    Subclasses implement: load_data, preprocess, train, evaluate.
    """

    def __init__(
        self,
        config_name: str,
        use_wandb: bool = True,
        modality: str | None = None,
    ):
        """Initialize trainer.

        Args:
            config_name: Problem name matching a YAML in ``configs/``.
            use_wandb: Enable W&B logging when an API key is available.
            modality: Optional data modality ("synthetic" | "stream" | "mixed").
                When set, the checkpoint dir becomes
                ``checkpoints/<problem>_<modality>/`` and the modality is
                recorded in metadata + MLflow tags. When None, the legacy
                ``checkpoints/<problem>/`` path is used (backward compatible).
        """
        self.config = load_config(config_name)
        self.problem = self.config["problem"]
        self.modality = modality
        self.use_wandb = use_wandb and self._init_wandb()
        self.use_mlflow = self._init_mlflow()
        self.metrics = {}
        self.model = None
        self.start_time = None

    def _init_wandb(self) -> bool:
        """Initialize W&B if API key is available."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key or api_key == "your_key_here":
            logger.info("W&B API key not set. Using local logging only.")
            return False

        try:
            import wandb

            wandb.init(
                project=self.config.get("training", {}).get("wandb_project", "portfolio-ml-system"),
                name=f"{self.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                tags=self.config.get("training", {}).get("wandb_tags", []),
            )
            logger.info("W&B initialized successfully.")
            return True
        except Exception as e:
            logger.warning("W&B init failed: %s. Using local logging.", e)
            return False

    def _init_mlflow(self) -> bool:
        """Initialize MLflow tracking. Falls back gracefully if server unreachable."""
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = self.config.get(
                "training", {},
            ).get("wandb_project", "portfolio-ml-system")
            mlflow.set_experiment(experiment_name)

            suffix = f"_{self.modality}" if self.modality else ""
            run_name = (
                f"{self.problem}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            # Auto-nest when a parent MLflow run is already active (e.g. under
            # `--modality all` which creates a comparison parent run in the CLI).
            is_nested = mlflow.active_run() is not None
            mlflow.start_run(run_name=run_name, nested=is_nested)

            # Tag the run with modality so MLflow UI can group/filter.
            if self.modality:
                mlflow.set_tag("modality", self.modality)

            # Log config as flat params
            self._log_config_as_params(self.config)
            logger.info(
                "MLflow tracking initialized (nested=%s, modality=%s).",
                is_nested,
                self.modality,
            )
            return True
        except Exception as e:
            logger.warning("MLflow init failed: %s. Continuing without MLflow.", e)
            return False

    def _log_config_as_params(self, config: dict, prefix: str = "") -> None:
        """Flatten nested config dict and log as MLflow params."""
        params = {}
        for key, value in config.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict):
                self._log_config_as_params(value, full_key)
            elif isinstance(value, list):
                params[full_key] = str(value)
            else:
                params[full_key] = str(value)
        if params:
            mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to W&B, MLflow, and local storage."""
        self.metrics[key] = value
        if self.use_wandb:
            import wandb

            wandb.log({key: value}, step=step)
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log multiple metrics."""
        self.metrics.update(metrics)
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

    def get_checkpoint_dir(self) -> Path:
        """Return the checkpoint directory for this trainer.

        When ``self.modality`` is set, path is
        ``checkpoints/<problem>_<modality>/`` so three-modality runs don't
        clobber each other. When None, uses the legacy ``training.checkpoint_dir``
        config value (or ``checkpoints/<problem>/`` default) for backward
        compatibility with the original 4 models.
        """
        if self.modality:
            return (
                get_project_root() / "checkpoints" / f"{self.problem}_{self.modality}"
            )
        configured = self.config.get("training", {}).get(
            "checkpoint_dir",
            str(get_project_root() / "checkpoints" / self.problem),
        )
        path = Path(configured)
        return path if path.is_absolute() else get_project_root() / path

    def save_checkpoint(self, model_artifacts: dict) -> Path:
        """Save model checkpoint and metadata.

        Args:
            model_artifacts: Dict of {filename: save_fn} where save_fn(path) saves the artifact.

        Returns:
            Checkpoint directory path.
        """
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifacts
        for filename, save_fn in model_artifacts.items():
            filepath = checkpoint_dir / filename
            save_fn(filepath)
            logger.info("Saved artifact: %s", filepath)

        # MLflow: log artifacts and register model
        mlflow_run_id = None
        mlflow_model_version = None
        if self.use_mlflow and mlflow.active_run():
            try:
                mlflow_run_id = mlflow.active_run().info.run_id
                mlflow.log_artifacts(str(checkpoint_dir))

                client = mlflow.tracking.MlflowClient()
                try:
                    client.create_registered_model(self.problem)
                except mlflow.exceptions.MlflowException:
                    pass  # Already exists
                mv = client.create_model_version(
                    name=self.problem,
                    source=f"runs:/{mlflow_run_id}",
                    run_id=mlflow_run_id,
                )
                mlflow_model_version = mv.version
                logger.info(
                    "Registered model '%s' v%s in MLflow", self.problem, mlflow_model_version
                )
            except Exception as e:
                logger.warning("MLflow registry failed: %s", e)

        # Save metadata
        elapsed = time.time() - self.start_time if self.start_time else 0
        metadata = {
            "problem": self.problem,
            "model_type": self.config.get("model", {}).get("type", "unknown"),
            "modality": self.modality,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "hyperparameters": self.config.get("model", {}).get("params", {}),
            "training_time_seconds": round(elapsed, 1),
            "config_file": f"configs/{self.problem}.yaml",
            "mlflow_run_id": mlflow_run_id,
            "mlflow_model_version": mlflow_model_version,
        }

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved checkpoint metadata: %s", metadata_path)

        # Phase A.1.8 — dual-write: mirror the synthetic modality to the legacy
        # checkpoints/<problem>/ path so the existing ModelPredictor keeps
        # working unchanged. Stream / mixed modalities never touch the legacy
        # dir. Legacy mirror is retired in Phase A.9 when the predictor moves
        # to _<modality>/ paths directly.
        if self.modality == "synthetic":
            self._mirror_to_legacy_path(checkpoint_dir)

        return checkpoint_dir

    def _mirror_to_legacy_path(self, modality_dir: Path) -> None:
        """Copy ``checkpoints/<problem>_synthetic/`` to ``checkpoints/<problem>/``.

        Non-destructive ``copytree(..., dirs_exist_ok=True)`` — fully replaces
        any previously-mirrored artifacts so bytes stay in sync. Logs a
        deprecation notice pointing at Phase A.9 (predictor migration).
        """
        legacy_dir = get_project_root() / "checkpoints" / self.problem
        shutil.copytree(modality_dir, legacy_dir, dirs_exist_ok=True)
        logger.info(
            "Mirrored synthetic checkpoint %s -> legacy %s "
            "(deprecation: legacy path will be removed in Phase A.9; "
            "predictor should migrate to _<modality>/ paths)",
            modality_dir,
            legacy_dir,
        )

    def save_results_csv(self) -> Path:
        """Save evaluation metrics to CSV."""
        results_dir = get_project_root() / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        rows = [{"metric": k, "value": v} for k, v in self.metrics.items()]
        df = pd.DataFrame(rows)

        results_path = results_dir / f"{self.problem}_metrics.csv"
        df.to_csv(results_path, index=False)
        logger.info("Saved results CSV: %s", results_path)
        return results_path

    def finish(self) -> None:
        """Finalize W&B and MLflow runs."""
        if self.use_wandb:
            import wandb

            wandb.finish()
        if self.use_mlflow and mlflow.active_run():
            mlflow.end_run()

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw data."""
        ...

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> dict:
        """Feature engineering and data splitting. Returns dict of processed data."""
        ...

    @abstractmethod
    def train(self, data: dict) -> None:
        """Train the model. Sets self.model."""
        ...

    @abstractmethod
    def evaluate(self, data: dict) -> dict:
        """Evaluate the model. Returns metrics dict."""
        ...

    @abstractmethod
    def get_checkpoint_artifacts(self) -> dict:
        """Return {filename: save_fn} for checkpoint saving."""
        ...

    def run(self) -> dict:
        """Execute the full training pipeline."""
        setup_logging()
        logger.info("=" * 60)
        logger.info("Training: %s", self.problem)
        logger.info("=" * 60)

        self.start_time = time.time()

        # Load data
        logger.info("1. Loading data...")
        df = self.load_data()
        logger.info("   Loaded %d rows", len(df))

        # Preprocess
        logger.info("2. Preprocessing...")
        data = self.preprocess(df)

        # Train
        logger.info("3. Training model...")
        self.train(data)

        # Evaluate
        logger.info("4. Evaluating...")
        metrics = self.evaluate(data)
        self.log_metrics(metrics)

        # Save
        logger.info("5. Saving checkpoint and results...")
        self.save_checkpoint(self.get_checkpoint_artifacts())
        self.save_results_csv()

        elapsed = time.time() - self.start_time
        logger.info("Completed %s in %.1fs", self.problem, elapsed)
        for k, v in metrics.items():
            logger.info(
                "   %s: %.4f" if isinstance(v, float) else "   %s: %s",
                k,
                v,
            )

        self.finish()
        return metrics
