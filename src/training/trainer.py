"""Base trainer with W&B + MLflow integration and checkpointing."""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from rich.console import Console

from src.config import get_project_root, load_config

console = Console()


class BaseTrainer(ABC):
    """Base class for all model trainers.

    Handles: config loading, W&B + MLflow initialization, checkpoint saving, results CSV export.
    Subclasses implement: load_data, preprocess, train, evaluate.
    """

    def __init__(self, config_name: str, use_wandb: bool = True):
        self.config = load_config(config_name)
        self.problem = self.config["problem"]
        self.use_wandb = use_wandb and self._init_wandb()
        self.use_mlflow = self._init_mlflow()
        self.metrics = {}
        self.model = None
        self.start_time = None

    def _init_wandb(self) -> bool:
        """Initialize W&B if API key is available."""
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key or api_key == "your_key_here":
            console.print("[yellow]W&B API key not set. Using local logging only.[/yellow]")
            return False

        try:
            import wandb

            wandb.init(
                project=self.config.get("training", {}).get("wandb_project", "portfolio-ml-system"),
                name=f"{self.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                tags=self.config.get("training", {}).get("wandb_tags", []),
            )
            console.print("[green]W&B initialized successfully.[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]W&B init failed: {e}. Using local logging.[/yellow]")
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
            run_name = f"{self.problem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)

            # Log config as flat params
            self._log_config_as_params(self.config)
            console.print("[green]MLflow tracking initialized.[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]MLflow init failed: {e}. Continuing without MLflow.[/yellow]")
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

    def save_checkpoint(self, model_artifacts: dict) -> Path:
        """Save model checkpoint and metadata.

        Args:
            model_artifacts: Dict of {filename: save_fn} where save_fn(path) saves the artifact.

        Returns:
            Checkpoint directory path.
        """
        checkpoint_dir = Path(
            self.config.get("training", {}).get(
                "checkpoint_dir",
                str(get_project_root() / "checkpoints" / self.problem),
            )
        )
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = get_project_root() / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifacts
        for filename, save_fn in model_artifacts.items():
            filepath = checkpoint_dir / filename
            save_fn(filepath)
            console.print(f"  [green]Saved {filepath}[/green]")

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
                console.print(
                    f"  [green]Registered model '{self.problem}' "
                    f"v{mlflow_model_version} in MLflow[/green]"
                )
            except Exception as e:
                console.print(
                    f"  [yellow]MLflow registry failed: {e}[/yellow]"
                )

        # Save metadata
        elapsed = time.time() - self.start_time if self.start_time else 0
        metadata = {
            "problem": self.problem,
            "model_type": self.config.get("model", {}).get("type", "unknown"),
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
        console.print(f"  [green]Saved {metadata_path}[/green]")

        return checkpoint_dir

    def save_results_csv(self) -> Path:
        """Save evaluation metrics to CSV."""
        results_dir = get_project_root() / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        rows = [{"metric": k, "value": v} for k, v in self.metrics.items()]
        df = pd.DataFrame(rows)

        results_path = results_dir / f"{self.problem}_metrics.csv"
        df.to_csv(results_path, index=False)
        console.print(f"  [green]Saved results -> {results_path}[/green]")
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
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Training: {self.problem}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")

        self.start_time = time.time()

        # Load data
        console.print("\n[bold]1. Loading data...[/bold]")
        df = self.load_data()
        console.print(f"   Loaded {len(df):,} rows")

        # Preprocess
        console.print("\n[bold]2. Preprocessing...[/bold]")
        data = self.preprocess(df)

        # Train
        console.print("\n[bold]3. Training model...[/bold]")
        self.train(data)

        # Evaluate
        console.print("\n[bold]4. Evaluating...[/bold]")
        metrics = self.evaluate(data)
        self.log_metrics(metrics)

        # Save
        console.print("\n[bold]5. Saving checkpoint and results...[/bold]")
        self.save_checkpoint(self.get_checkpoint_artifacts())
        self.save_results_csv()

        elapsed = time.time() - self.start_time
        console.print(f"\n[bold green]Completed {self.problem} in {elapsed:.1f}s[/bold green]")
        for k, v in metrics.items():
            console.print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

        self.finish()
        return metrics
