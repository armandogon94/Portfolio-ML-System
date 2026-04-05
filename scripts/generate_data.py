"""CLI script to generate synthetic datasets."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.config import get_project_root
from src.data.generate_credit_risk import generate_credit_risk_data
from src.data.generate_fraud import generate_fraud_data
from src.data.generate_housing import generate_housing_data
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
