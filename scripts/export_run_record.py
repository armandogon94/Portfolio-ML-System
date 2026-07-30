#!/usr/bin/env python
"""Export safe committed run provenance from a real local checkpoint.

The checkpoint and row-level predictions remain gitignored. This command writes
only run metadata, aggregate metrics, source integrity fields, and aggregate
dataset counts to ``reports/<run>_run.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_project_root, load_config
from src.training.publication import write_run_record


def _summary_from_oof(path: Path, n_features: int) -> dict:
    frame = pd.read_csv(path, usecols=["y_true"])
    labels = frame["y_true"]
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path} contains labels outside 0 and 1.")
    return {
        "n_rows": int(len(frame)),
        "n_features": int(n_features),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
    }


def _summary_from_source(config: dict, data_path: Path | None, n_features: int) -> dict:
    adapter = importlib.import_module(config["data"]["source"]["adapter"])
    frame = adapter.load(path=data_path) if data_path is not None else adapter.load()
    target = config["data"]["target"]
    return {
        "n_rows": int(len(frame)),
        "n_features": int(n_features),
        "positive_count": int(frame[target].sum()),
        "positive_rate": float(frame[target].mean()),
    }


def export(model: str, *, data_path: Path | None = None) -> Path:
    root = get_project_root()
    metadata_path = root / "checkpoints" / model / "metadata.json"
    metrics_path = root / "reports" / f"{model}_metrics.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is missing. Train the real run before publishing it."
        )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} is missing.")

    metadata = json.loads(metadata_path.read_text())
    config = load_config(model)
    adapter = importlib.import_module(config["data"]["source"]["adapter"])
    metadata["source_integrity"] = {
        key: adapter.PROVENANCE.get(key)
        for key in ("expected_rows", "expected_sha256")
        if adapter.PROVENANCE.get(key) is not None
    }
    if "dataset_summary" not in metadata:
        oof_path = root / "reports" / f"{model}_oof_predictions.csv"
        if oof_path.is_file():
            metadata["dataset_summary"] = _summary_from_oof(oof_path, int(metadata["n_features"]))
        else:
            metadata["dataset_summary"] = _summary_from_source(
                config, data_path, int(metadata["n_features"])
            )

    destination = root / "reports" / f"{model}_run.json"
    return write_run_record(metadata, metrics_path, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("fraud_ulb", "credit_risk", "churn"), required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Optional existing source file for a non-CV run. The path is not recorded.",
    )
    args = parser.parse_args()
    path = export(args.model, data_path=args.data_path)
    print(path.relative_to(get_project_root()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
