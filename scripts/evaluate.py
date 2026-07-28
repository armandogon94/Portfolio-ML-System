#!/usr/bin/env python
"""Read the committed metrics CSVs and print the results table.

This script **never computes a metric**. It reads ``reports/*_metrics.csv``, which
are written by training runs, and renders them. That is deliberate: it is
structurally impossible for a number to appear in a table here that was not
produced by a training run, which is the failure mode this repository was rebuilt
to fix.

With ``--markdown`` it emits a compact one-row-per-problem metrics summary. The
README and ``reports/RESULTS.md`` use richer tables with dataset and per-model
context, so this output is evidence for their metric cells, not a paste-ready
replacement for either table.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import get_project_root

console = Console()
#: One row per *config*, not per problem: fraud has two datasets. Names match
#: ``configs/<name>.yaml`` and therefore ``reports/<name>_metrics.csv``.
RESULT_ROWS = (
    "fraud",
    "fraud_ulb",
    "fraud_autoencoder",
    "credit_risk",
    "churn",
)

#: Column order for the rendered table, matching README §Results.
COLUMNS = [
    ("pr_auc", "PR-AUC"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc_baseline", "Baseline PR-AUC"),
    ("pr_auc_delta", "Delta"),
    ("precision_at_1pct", "P@1%"),
    ("recall_at_1pct_fpr", "R@1%FPR"),
]


def _read(problem: str) -> dict[str, float]:
    """Return ``{metric: value}`` from a problem's committed CSV, or ``{}``."""
    path = get_project_root() / "reports" / f"{problem}_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["metric"], frame["value"]))


def _pick(metrics: dict[str, float], bare: str) -> str:
    """Prefer the CV mean +/- std form when present, else the single test value."""
    mean, std = metrics.get(f"cv_{bare}_mean"), metrics.get(f"cv_{bare}_std")
    if mean is not None:
        return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"
    value = metrics.get(f"test_{bare}")
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table body.")
    args = parser.parse_args()

    rows = {problem: _read(problem) for problem in RESULT_ROWS}
    measured = {p: m for p, m in rows.items() if m}

    if args.markdown:
        for problem in RESULT_ROWS:
            metrics = rows[problem]
            cells = [_pick(metrics, bare) or " " for bare, _ in COLUMNS]
            print(f"| {problem} | " + " | ".join(cells) + " |")
        if not measured:
            print("\n<!-- Every cell is empty because no model has been trained yet. -->")
        return 0

    table = Table(title="Results, read from reports/*_metrics.csv and never typed by hand")
    table.add_column("problem", style="cyan")
    for _, label in COLUMNS:
        table.add_column(label, justify="right")
    for problem in RESULT_ROWS:
        table.add_row(problem, *[_pick(rows[problem], bare) or "n/a" for bare, _ in COLUMNS])
    console.print(table)

    if not measured:
        console.print(
            "\n[yellow]No metrics CSV exists for any problem. Nothing has been "
            "trained on real data yet.[/yellow]\n"
            "  1. uv run python scripts/download_data.py --dataset all\n"
            "  2. uv run python scripts/train.py --model all\n"
            "See docs/PROGRESS.md for what is blocked and why."
        )
        return 0

    # Surface any leak warning the trainer recorded, so it cannot be missed.
    for problem in measured:
        metadata_path = get_project_root() / "checkpoints" / problem / "metadata.json"
        if not metadata_path.exists():
            continue
        warning = json.loads(metadata_path.read_text()).get("sanity_band_warning")
        if warning:
            console.print(f"\n[bold yellow]{problem}: {warning}[/bold yellow]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
