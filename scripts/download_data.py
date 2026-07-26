#!/usr/bin/env python
"""The single documented entrypoint for obtaining real data.

Downloads to ``~/.cache/kagglehub`` (outside the repository and outside the Docker
build context), records row counts and SHA-256, compares them when provenance is
pinned, and prints a block you can paste into ``data/README.md``.

On failure it prints the exact remediation — the URL to accept competition rules,
or where to put ``kaggle.json`` — and exits non-zero. **There is no synthetic
fallback.** If the real data cannot be obtained, nothing is trained and no number
is published. See ``docs/adr/0003-real-data-over-synthetic.md``.

Usage:
    uv run python scripts/download_data.py --dataset cc-churn
    uv run python scripts/download_data.py --dataset ieee-cis
    uv run python scripts/download_data.py --dataset lending-club
    uv run python scripts/download_data.py --dataset ulb-creditcard   # no credentials
    uv run python scripts/download_data.py --dataset all
    uv run python scripts/download_data.py --check                    # dry run, no download
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.adapters import credit_card_churn, ieee_cis, lending_club, ulb_creditcard
from src.data.download import (
    DatasetAccessError,
    kaggle_competition_cached,
    kaggle_dataset_cached,
    sha256_of,
)

console = Console()

#: ``--dataset`` name -> what to do about it.
#: ``approx_mb`` is the download size, stated so nobody is surprised by 648 MB.
DATASETS = {
    "ieee-cis": {
        "provenance": ieee_cis.PROVENANCE,
        "approx_mb": 118,
        "expanded_mb": 1350,
        "primary_file": "train_transaction.csv",
    },
    "lending-club": {
        "provenance": lending_club.PROVENANCE,
        # Measured 2026-07-26: the two .gz files are 648 MB together
        # (accepted 392.6 + rejected 255.5), kagglehub transfers ~1.26 GB, and
        # it also extracts each archive, leaving a ~3.94 GB cache directory.
        "approx_mb": 648,
        "expanded_mb": 3936,
        "primary_file": lending_club.PROVENANCE["filename"],
    },
    "cc-churn": {
        "provenance": credit_card_churn.PROVENANCE,
        "approx_mb": 2,
        "expanded_mb": 2,
        "primary_file": credit_card_churn.PROVENANCE["filename"],
    },
    "ulb-creditcard": {
        "provenance": ulb_creditcard.PROVENANCE,
        "approx_mb": 150,
        "expanded_mb": 150,
        "primary_file": None,
    },
}

_KAGGLE_KINDS = {"kaggle_competition", "kaggle_dataset"}


def _fetch(name: str) -> dict:
    """Download one dataset and return a provenance record for data/README.md."""
    spec = DATASETS[name]
    provenance = spec["provenance"]
    kind = provenance["kind"]

    console.print(f"\n[bold cyan]==> {name}[/bold cyan]  {provenance['name']}")
    console.print(f"    licence: {provenance['licence']}")
    console.print(f"    access:  {provenance['access']}")
    console.print(f"    size:    ~{spec['approx_mb']} MB download")

    if kind == "openml":
        frame = ulb_creditcard.load()
        record = {
            "dataset": name,
            **{k: provenance[k] for k in ("name", "url", "licence")},
            "rows": int(frame.shape[0]),
            "cols": int(frame.shape[1]),
            "positive_rate": float(frame[ulb_creditcard.TARGET].mean()),
            "sha256": "n/a",
        }
        expected_rows = provenance.get("expected_rows")
        if expected_rows is not None and len(frame) != expected_rows:
            console.print(
                "[bold yellow]    WARNING: ROW COUNT MISMATCH — "
                f"expected {expected_rows}, observed {len(frame)}. The vendor may "
                "have re-uploaded the frame; inspect it before training.[/bold yellow]"
            )
        console.print(
            f"[green]    verified {record['rows']} rows x {record['cols']} cols; "
            f"positive rate {record['positive_rate']:.6f}; sha256: n/a[/green]"
        )
        return record

    if kind == "kaggle_competition":
        location = Path(kaggle_competition_cached(provenance["slug"]))
    elif kind == "kaggle_dataset":
        location = Path(kaggle_dataset_cached(provenance["slug"]))
    else:  # pragma: no cover - guarded by the DATASETS table
        raise ValueError(f"Unknown source kind {kind!r} for {name}")

    record: dict = {
        "dataset": name,
        **{k: provenance[k] for k in ("name", "url", "licence")},
        "cache_path": str(location),
    }

    primary = spec["primary_file"]
    if primary:
        matches = sorted(location.rglob(primary))
        if not matches:
            raise DatasetAccessError(
                f"{primary!r} not found under {location}. The download may be partial; "
                f"delete {location} and retry."
            )

        target = matches[0]
        record["file"] = target.name
        record["size_mb"] = round(target.stat().st_size / 1e6, 1)
        console.print("    hashing (streamed, this takes a moment on large files)...")
        digest = sha256_of(target)
        record["sha256"] = digest

        expected_digest = provenance.get("expected_sha256")
        if expected_digest:
            if digest.casefold() != str(expected_digest).casefold():
                raise DatasetAccessError(
                    f"SHA-256 mismatch for {target.name}: expected {expected_digest}, "
                    f"observed {digest}. Delete the cache at {location} and retry; "
                    "do not train from a file whose provenance check failed."
                )
            console.print(f"[green]    SHA-256 matches pinned digest {expected_digest}[/green]")
        else:
            console.print(
                "[bold yellow]    RECORD THIS:[/bold yellow] "
                f'PROVENANCE["expected_sha256"] = "{digest}"'
            )

        rows = _count_csv_rows(target)
        record["rows"] = rows
        expected_rows = provenance.get("expected_rows")
        if expected_rows is not None and rows != expected_rows:
            console.print(
                "[bold yellow]    WARNING: ROW COUNT MISMATCH — "
                f"expected {expected_rows}, observed {rows}. The vendor may have "
                "re-uploaded the file; inspect it before training.[/bold yellow]"
            )

    console.print(f"[green]    cached at {location}[/green]")
    return record


def _count_csv_rows(path: Path) -> int:
    """Count data records in a CSV or CSV.gz without loading it into memory."""
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8", errors="replace", newline="") as handle:
        count = sum(1 for _ in csv.reader(handle))
    return max(0, count - 1)


def _check_only() -> int:
    """Report what each dataset needs, without downloading anything."""
    table = Table(title="Dataset access requirements (no download performed)")
    table.add_column("--dataset", style="cyan")
    table.add_column("size", justify="right")
    table.add_column("credentials")
    table.add_column("licence")
    for name, spec in DATASETS.items():
        provenance = spec["provenance"]
        needs = "none" if provenance["kind"] == "openml" else "Kaggle token"
        if provenance["kind"] == "kaggle_competition":
            needs += " + rules acceptance"
        table.add_row(name, f"{spec['approx_mb']} MB", needs, provenance["licence"])
    console.print(table)
    console.print(
        "\n[bold]ulb-creditcard needs no account at all[/bold] — a reviewer with zero "
        "Kaggle presence can still reproduce a real-data fraud result end to end."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print access requirements and exit without downloading.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to append the provenance records to, as JSON lines.",
    )
    args = parser.parse_args()

    if args.check or args.dataset is None:
        return _check_only()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    records, failures = [], []

    for name in names:
        try:
            records.append(_fetch(name))
        except DatasetAccessError as exc:
            console.print(f"[bold red]FAILED: {name}[/bold red]")
            console.print(str(exc))
            failures.append(name)
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected verbatim
            console.print(f"[bold red]FAILED: {name} ({type(exc).__name__})[/bold red] {exc}")
            failures.append(name)

    if records:
        console.print("\n[bold]Provenance — paste into data/README.md:[/bold]")
        console.print_json(json.dumps(records, indent=2))
        if args.out:
            with open(args.out, "a") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

    if failures:
        console.print(
            f"\n[bold red]{len(failures)} of {len(names)} datasets unavailable: "
            f"{failures}[/bold red]\nNo synthetic fallback is provided. "
            "See docs/adr/0003-real-data-over-synthetic.md."
        )
        return 1

    console.print(f"\n[bold green]{len(records)} dataset(s) ready.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
