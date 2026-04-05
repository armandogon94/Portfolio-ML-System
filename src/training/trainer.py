"""Base trainer with W&B integration and checkpointing."""

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.config import load_config, get_project_root

console = Console()


class BaseTrainer(ABC):
    """Base class for all model trainers.

    Handles: config loading, W&B initialization, checkpoint saving, results CSV export.
    Subclasses implement: load_data, preprocess, train, evaluate.
    """

    def __init__(self, config_name: str, use_wandb: bool = True):
        self.config = load_config(config_name)
        self.problem = self.config["problem"]
        self.use_wandb = use_wandb and self._init_wandb()
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

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to W&B and local storage."""
        self.metrics[key] = value
        if self.use_wandb:
            import wandb

            wandb.log({key: value}, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        """Log multiple metrics."""
        self.metrics.update(metrics)
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=step)

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
        """Finalize W&B run."""
        if self.use_wandb:
            import wandb

            wandb.finish()

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
