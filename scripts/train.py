"""CLI script to train models.

Phase A.1.9 adds ``--modality {synthetic,stream,mixed,all}``. When ``all``,
three training runs execute sequentially under one MLflow parent run; the
trainer's ``_init_mlflow`` auto-nests each child. After the loop we pick a
"recommended" modality by the per-model key metric and flag it in the
winning checkpoint's ``metadata.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import get_project_root, load_config

console = Console()

TRAINERS = {
    "credit_risk": "src.training.train_credit_risk:CreditRiskTrainer",
    "fraud": "src.training.train_fraud:FraudDetectionTrainer",
    "price": "src.training.train_price:PricePredictionTrainer",
    "forecaster": "src.training.train_forecaster:DemandForecastTrainer",
    "customer_churn": "src.training.train_customer_churn:CustomerChurnTrainer",
}

# CLI model name → problem / config name used by BaseTrainer.
CONFIG_NAMES = {
    "credit_risk": "credit_risk",
    "fraud": "fraud_detection",
    "price": "price_prediction",
    "forecaster": "demand_forecasting",
    "customer_churn": "customer_churn",
}

MODALITY_CHOICES = ("synthetic", "stream", "mixed", "all")
_ITER_MODALITIES = ("synthetic", "stream", "mixed")

# Which metric ranks modalities per model, and whether higher is better.
# Used to pick the "recommended" modality after --modality all.
RECOMMENDATION_KEY: dict[str, tuple[str, bool]] = {
    "price": ("test_r2", True),
    "credit_risk": ("test_auc_roc", True),
    "fraud": ("test_autoencoder_auc_roc", True),
    "forecaster": ("test_avg_mae", False),
    "customer_churn": ("test_auc_roc", True),
}


def get_trainer_class(name: str):
    module_path, class_name = TRAINERS[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _supports_modalities(model_name: str) -> bool:
    """A model supports modalities iff its config declares a kaggle_slug."""
    config = load_config(CONFIG_NAMES[model_name])
    return bool(config.get("data", {}).get("kaggle_slug"))


def _pick_recommended(
    model_name: str, per_modality: dict[str, dict]
) -> str | None:
    """Return the modality whose key metric wins, or None if none reported it."""
    spec = RECOMMENDATION_KEY.get(model_name)
    if spec is None:
        return None
    metric_name, higher_better = spec

    scored = {
        m: metrics.get(metric_name)
        for m, metrics in per_modality.items()
        if isinstance(metrics.get(metric_name), (int, float))
    }
    if not scored:
        return None
    return max(scored, key=scored.get) if higher_better else min(
        scored, key=scored.get
    )


def _flag_recommended_metadata(model_name: str, modality: str) -> None:
    """Write ``recommended=True`` into the winning checkpoint's metadata.json."""
    problem = CONFIG_NAMES[model_name]
    metadata_path = (
        get_project_root()
        / "checkpoints"
        / f"{problem}_{modality}"
        / "metadata.json"
    )
    if not metadata_path.exists():
        return
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["recommended"] = True
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _write_comparison_csv(
    model_name: str, per_modality: dict[str, dict], recommended: str | None
) -> Path:
    """Write results/modality_comparison_<problem>.csv — 3 rows, wide format."""
    problem = CONFIG_NAMES[model_name]
    rows = []
    for modality in _ITER_MODALITIES:
        metrics = per_modality.get(modality, {})
        row = {"modality": modality}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        row["recommended"] = modality == recommended
        rows.append(row)

    out = get_project_root() / "results" / f"modality_comparison_{problem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _print_comparison_table(
    model_name: str, per_modality: dict[str, dict], recommended: str | None
) -> None:
    spec = RECOMMENDATION_KEY.get(model_name)
    metric_name = spec[0] if spec else "test_accuracy"

    table = Table(title=f"{model_name} — modality comparison")
    table.add_column("modality", style="cyan")
    table.add_column(metric_name)
    table.add_column("recommended", justify="center")

    for modality in _ITER_MODALITIES:
        metrics = per_modality.get(modality, {})
        val = metrics.get(metric_name)
        val_str = f"{val:.4f}" if isinstance(val, float) else (str(val) if val is not None else "—")
        table.add_row(modality, val_str, "✓" if modality == recommended else "")

    console.print()
    console.print(table)
    if recommended:
        console.print(
            f"[bold green]Recommended for demo: {recommended}[/bold green]"
        )


def _run_modality_all(model_name: str, use_wandb: bool) -> dict[str, dict]:
    """Train all 3 modalities under one MLflow parent run and write comparison."""
    import mlflow

    TrainerClass = get_trainer_class(model_name)
    per_modality: dict[str, dict] = {}

    parent_name = (
        f"{model_name}_modality_comparison_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    with mlflow.start_run(run_name=parent_name):
        mlflow.set_tag("modality", "all")
        for modality in _ITER_MODALITIES:
            console.print(
                f"\n[bold cyan]══ Training {model_name} ({modality}) ══[/bold cyan]"
            )
            trainer = TrainerClass(use_wandb=use_wandb, modality=modality)
            per_modality[modality] = trainer.run()

    recommended = _pick_recommended(model_name, per_modality)
    if recommended:
        _flag_recommended_metadata(model_name, recommended)

    comparison_path = _write_comparison_csv(model_name, per_modality, recommended)
    console.print(f"[dim]Comparison CSV: {comparison_path}[/dim]")
    _print_comparison_table(model_name, per_modality, recommended)

    return per_modality


def _train_one(model_name: str, modality: str | None, use_wandb: bool) -> dict:
    TrainerClass = get_trainer_class(model_name)
    trainer = TrainerClass(use_wandb=use_wandb, modality=modality)
    return trainer.run()


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        required=True,
        help="Which model to train",
    )
    parser.add_argument(
        "--modality",
        choices=MODALITY_CHOICES,
        default=None,
        help=(
            "Data modality: synthetic | stream | mixed | all. "
            "With 'all', trains all three under one MLflow parent run and "
            "writes a comparison CSV. Models without kaggle_slug in their "
            "config fall back to legacy single-run behavior."
        ),
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    models = list(TRAINERS.keys()) if args.model == "all" else [args.model]
    use_wandb = not args.no_wandb

    console.print("[bold]Model Training Pipeline[/bold]")
    all_metrics: dict[str, dict] = {}

    for model_name in models:
        if args.modality == "all":
            if _supports_modalities(model_name):
                all_metrics[model_name] = _run_modality_all(model_name, use_wandb)
                continue
            console.print(
                f"[yellow]{model_name}: no kaggle_slug in config; "
                f"ignoring --modality all, running legacy single train.[/yellow]"
            )
            all_metrics[model_name] = _train_one(model_name, None, use_wandb)
        elif args.modality in _ITER_MODALITIES:
            all_metrics[model_name] = _train_one(model_name, args.modality, use_wandb)
        else:
            # --modality not provided → legacy behavior (modality=None)
            all_metrics[model_name] = _train_one(model_name, None, use_wandb)

    console.print("\n[bold green]All models trained successfully![/bold green]")


if __name__ == "__main__":
    main()
