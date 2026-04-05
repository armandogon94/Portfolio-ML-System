"""CLI script to train models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

console = Console()

TRAINERS = {
    "credit_risk": "src.training.train_credit_risk:CreditRiskTrainer",
    "fraud": "src.training.train_fraud:FraudDetectionTrainer",
    "price": "src.training.train_price:PricePredictionTrainer",
    "forecaster": "src.training.train_forecaster:DemandForecastTrainer",
}


def get_trainer_class(name: str):
    module_path, class_name = TRAINERS[name].rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()) + ["all"],
        required=True,
        help="Which model to train",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    models = list(TRAINERS.keys()) if args.model == "all" else [args.model]
    use_wandb = not args.no_wandb

    console.print("[bold]Model Training Pipeline[/bold]")
    all_metrics = {}

    for model_name in models:
        TrainerClass = get_trainer_class(model_name)
        trainer = TrainerClass(use_wandb=use_wandb)
        metrics = trainer.run()
        all_metrics[model_name] = metrics

    console.print("\n[bold green]All models trained successfully![/bold green]")

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    for name, metrics in all_metrics.items():
        key_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        top_metrics = dict(list(key_metrics.items())[:3])
        formatted = ", ".join(f"{k}: {v:.4f}" for k, v in top_metrics.items())
        console.print(f"  [cyan]{name}[/cyan]: {formatted}")


if __name__ == "__main__":
    main()
