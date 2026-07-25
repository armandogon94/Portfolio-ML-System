#!/usr/bin/env python
"""Train one problem, or all three, from its config.

There is no ``--modality`` flag. The old three-modality
(synthetic / stream / mixed) machinery was deleted: real data is now the only
path. See ``docs/adr/0003-real-data-over-synthetic.md``.

Usage:
    uv run python scripts/train.py --model churn
    uv run python scripts/train.py --model fraud --autoencoder
    uv run python scripts/train.py --model all
    uv run python scripts/train.py --model churn --sample     # CI fixture, no checkpoint

Requires data. Run ``scripts/download_data.py`` first; without it this exits with
the exact remediation rather than falling back to anything synthetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.config import PROBLEMS
from src.data.download import DatasetAccessError

console = Console()

#: Metrics worth printing after a run. Everything else lands in the CSV.
_HEADLINE = (
    "test_pr_auc",
    "test_roc_auc",
    "test_pr_auc_baseline",
    "test_pr_auc_delta",
    "test_precision_at_1pct",
    "test_recall_at_1pct_fpr",
    "cv_pr_auc_mean",
    "cv_pr_auc_std",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
)


def _summary_table(results: dict[str, dict[str, float]]) -> Table:
    table = Table(title="Measured metrics (also written to reports/<problem>_metrics.csv)")
    table.add_column("problem", style="cyan")
    for name in _HEADLINE:
        table.add_column(name.replace("test_", "").replace("cv_", "cv "), justify="right")

    for problem, metrics in results.items():
        row = [problem]
        for name in _HEADLINE:
            value = metrics.get(name)
            row.append(f"{value:.4f}" if isinstance(value, float) else "—")
        table.add_row(*row)
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=[*PROBLEMS, "all"], required=True)
    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="Train the unsupervised MPS autoencoder baseline instead (fraud only).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Train on the committed CI fixtures. Writes NO checkpoint and NO metrics "
            "CSV — fixture numbers must never become published numbers."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help="Also log to W&B if a key is set.")
    args = parser.parse_args()

    if args.autoencoder and args.model not in ("fraud", "all"):
        parser.error("--autoencoder applies to the fraud problem only.")

    problems = list(PROBLEMS) if args.model == "all" else [args.model]
    results: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for problem in problems:
        console.rule(f"[bold cyan]{problem}")
        try:
            trainer = _build_trainer(problem, args)
            results[trainer_label(problem, args)] = trainer.run()
        except DatasetAccessError as exc:
            console.print(f"[bold red]{problem}: data unavailable[/bold red]\n{exc}")
            failures.append(problem)
        except FileNotFoundError as exc:
            console.print(f"[bold red]{problem}: {exc}[/bold red]")
            failures.append(problem)

    if results:
        console.print()
        console.print(_summary_table(results))
    if args.sample:
        console.print(
            "\n[yellow]SAMPLE MODE — nothing was checkpointed. These numbers are "
            "from synthetic fixtures and are meaningless as model results.[/yellow]"
        )

    if failures:
        console.print(
            f"\n[bold red]{len(failures)} problem(s) not trained: {failures}[/bold red]\n"
            "Fetch the data first:  uv run python scripts/download_data.py --dataset all"
        )
        return 1
    return 0


def trainer_label(problem: str, args: argparse.Namespace) -> str:
    return f"{problem}_autoencoder" if args.autoencoder and problem == "fraud" else problem


def _build_trainer(problem: str, args: argparse.Namespace):
    if args.autoencoder and problem == "fraud":
        from src.training.autoencoder_pipeline import AutoencoderTrainer

        return AutoencoderTrainer(problem, use_wandb=args.wandb, sample=args.sample)

    from src.training.tabular import TabularTrainer

    return TabularTrainer(problem, use_wandb=args.wandb, sample=args.sample)


if __name__ == "__main__":
    raise SystemExit(main())
