#!/usr/bin/env python
"""Describe a config's split: partition boundaries, sizes, and positive rates.

Every claim a results document makes about *how* a dataset was divided should be
regenerable rather than remembered. This script loads a problem exactly as
``scripts/train.py`` does (same adapter, same config, same split function) and
prints the resulting partition geometry. It trains nothing and writes no metric.

It exists because a time-ordered split on real data is never as clean as the
config implies. ``issue_d`` is monthly, so honouring timestamp ties moves the
requested 70/10/20 boundary to the nearest month edge; and filtering LendingClub
to terminal statuses before splitting leaves the latest vintages enriched for
loans that resolved unusually fast. Both effects are visible in this output and
neither is visible in an accuracy number.

Usage:
    uv run python scripts/describe_split.py --model credit_risk
    uv run python scripts/describe_split.py --model credit_risk --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import available_config_names, load_config
from src.data.adapters import get_adapter
from src.data.split import Split, make_splits

console = Console()


def _partition_record(
    name: str, frame: pd.DataFrame, indices: np.ndarray, column: str | None, target: str
) -> dict:
    """Summarise one partition: size, positive rate, and time span."""
    record: dict = {"partition": name, "n": int(len(indices))}
    if len(indices) == 0:
        return record

    rows = frame.iloc[indices]
    record["positive_rate"] = float(rows[target].mean())
    record["n_positive"] = int(rows[target].sum())
    if column and column in frame.columns:
        values = rows[column]
        record["min"] = str(values.min())
        record["max"] = str(values.max())
    return record


def _overlap(frame: pd.DataFrame, splits: list[Split], column: str) -> list[str]:
    """Return split-key values that appear in more than one partition.

    A time-ordered split that lets one timestamp straddle a boundary is not a
    time-ordered split. This is the assertion, run against the real data rather
    than against a fixture of unique integers.
    """
    named = {"train": splits[0].train, "val": splits[0].val, "test": splits[0].test}
    seen: dict[str, set] = {}
    for name, indices in named.items():
        if len(indices):
            seen[name] = set(frame.iloc[indices][column].unique())

    shared: set = set()
    names = list(seen)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared |= seen[left] & seen[right]
    return sorted(str(value) for value in shared)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(available_config_names()), required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    config = load_config(args.model)
    target = config["data"]["target"]
    column = config["split"].get("column")

    adapter = get_adapter(config["data"]["source"]["adapter"])
    frame = adapter.load()
    splits = make_splits(frame, config["split"], target, config["seed"])

    report: dict = {
        "config": f"configs/{args.model}.yaml",
        "split_type": config["split"]["type"],
        "split_column": column,
        "n_rows": int(len(frame)),
        "overall_positive_rate": float(frame[target].mean()),
        "n_splits": len(splits),
        "partitions": [],
    }

    if len(splits) == 1:
        split = splits[0]
        for name, indices in (
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ):
            report["partitions"].append(_partition_record(name, frame, indices, column, target))
        if column:
            shared = _overlap(frame, splits, column)
            report["split_key_values_in_multiple_partitions"] = shared
            report["split_key_leak"] = bool(shared)
    else:
        for index, split in enumerate(splits, start=1):
            report["partitions"].append(
                _partition_record(f"fold{index}_test", frame, split.test, column, target)
            )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    console.print(
        f"[bold]{report['config']}[/bold]: {report['split_type']} split on "
        f"{column!r}; {report['n_rows']:,} rows, overall positive rate "
        f"{report['overall_positive_rate']:.4f}"
    )
    table = Table(title="Partitions")
    table.add_column("partition", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("positives", justify="right")
    table.add_column("positive rate", justify="right")
    table.add_column(f"min {column}", justify="right")
    table.add_column(f"max {column}", justify="right")
    for record in report["partitions"]:
        table.add_row(
            record["partition"],
            f"{record['n']:,}",
            f"{record.get('n_positive', 0):,}",
            f"{record.get('positive_rate', float('nan')):.4f}",
            record.get("min", "n/a"),
            record.get("max", "n/a"),
        )
    console.print(table)

    if "split_key_leak" in report:
        if report["split_key_leak"]:
            shared = report["split_key_values_in_multiple_partitions"]
            console.print(
                f"[bold red]SPLIT KEY LEAK: {len(shared)} value(s) of {column!r} "
                f"appear in more than one partition.[/bold red]"
            )
            return 1
        console.print(f"[green]No value of {column!r} appears in more than one partition.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
