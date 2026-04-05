"""CLI script to evaluate trained models and generate comparison summary."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from src.evaluation.evaluator import save_comparison_summary

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models and create summary")
    parser.add_argument("--model", default="all", help="Which model to evaluate (default: all)")
    args = parser.parse_args()

    summary_path = save_comparison_summary()
    console.print(f"[bold green]Comparison summary saved -> {summary_path}[/bold green]")


if __name__ == "__main__":
    main()
