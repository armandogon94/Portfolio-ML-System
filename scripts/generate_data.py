"""CLI script to generate synthetic datasets."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.config import get_project_root
from src.data.generate_credit_risk import generate_credit_risk_data
from src.data.generate_customer_churn import generate_customer_churn_data
from src.data.generate_delivery_eta import generate_delivery_eta_data
from src.data.generate_dental_noshow import generate_dental_noshow_data
from src.data.generate_fraud import generate_fraud_data
from src.data.generate_h1b_approval import generate_h1b_approval_data
from src.data.generate_heart_disease import generate_heart_disease_data
from src.data.generate_housing import generate_housing_data
from src.data.generate_rental_price import generate_rental_price_data
from src.data.generate_timeseries import generate_timeseries_data

console = Console()


GENERATORS = {
    "credit_risk": {
        "fn": generate_credit_risk_data,
        "output": "credit_risk.csv",
        "kwargs": {"n_samples": 50000},
    },
    "fraud": {
        "fn": generate_fraud_data,
        "output": "fraud_transactions.csv",
        "kwargs": {"n_samples": 200000},
    },
    "housing": {
        "fn": generate_housing_data,
        "output": "housing.csv",
        "kwargs": {"n_samples": 30000},
    },
    "timeseries": {
        "fn": generate_timeseries_data,
        "output": "daily_demand.csv",
        "kwargs": {"n_years": 3},
    },
    "rental_price": {
        "fn": generate_rental_price_data,
        "output": "rental_price.csv",
        "kwargs": {"n_samples": 5000},
    },
    "dental_noshow": {
        "fn": generate_dental_noshow_data,
        "output": "dental_noshow.csv",
        "kwargs": {"n_samples": 5000},
    },
    "heart_disease": {
        "fn": generate_heart_disease_data,
        "output": "heart_disease.csv",
        "kwargs": {"n_samples": 20000},
    },
    "delivery_eta": {
        "fn": generate_delivery_eta_data,
        "output": "delivery_eta.csv",
        "kwargs": {"n_samples": 30000},
    },
    "customer_churn": {
        "fn": generate_customer_churn_data,
        "output": "customer_churn.csv",
        "kwargs": {"n_samples": 20000},
    },
    "h1b_approval": {
        "fn": generate_h1b_approval_data,
        "output": "h1b_approval.csv",
        "kwargs": {"n_samples": 50000},
    },
}


def generate(problem: str) -> None:
    gen = GENERATORS[problem]
    console.print(f"\n[bold blue]Generating {problem} data...[/bold blue]")

    df = gen["fn"](**gen["kwargs"])

    output_dir = get_project_root() / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / gen["output"]

    df.to_csv(output_path, index=False)
    console.print(f"  [green]Saved {len(df):,} rows -> {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets")
    parser.add_argument(
        "--problem",
        choices=list(GENERATORS.keys()) + ["all"],
        required=True,
        help="Which dataset to generate",
    )
    args = parser.parse_args()

    problems = list(GENERATORS.keys()) if args.problem == "all" else [args.problem]

    console.print("[bold]Synthetic Data Generation[/bold]")
    for problem in problems:
        generate(problem)

    console.print("\n[bold green]All datasets generated successfully![/bold green]")


if __name__ == "__main__":
    main()
