"""Write a safe, committed provenance record for every published training run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

_PUBLIC_METADATA_KEYS = (
    "problem",
    "model_type",
    "seed",
    "git_sha",
    "source_tree_sha256",
    "worktree_dirty_at_training",
    "trained_at",
    "dataset",
    "dataset_summary",
    "source_integrity",
    "split",
    "n_features",
    "hyperparameters",
    "config_file",
    "mlflow_run_id",
    "checkpoint_fit",
    "evaluation_predictions",
    "hardware",
    "leakage_controls",
    "sanity_band_warning",
)


def read_metrics_csv(path: Path) -> dict[str, float]:
    """Read one training metrics CSV and reject ambiguous or non-finite rows."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if set(frame.columns) != {"metric", "value"}:
        raise ValueError(f"{path} must contain exactly metric and value columns.")
    duplicates = sorted(frame.loc[frame["metric"].duplicated(keep=False), "metric"].unique())
    if duplicates:
        raise ValueError(f"{path} contains duplicate metric rows: {duplicates}.")
    if frame["metric"].isna().any() or (frame["metric"].astype(str).str.strip() == "").any():
        raise ValueError(f"{path} contains an empty metric name.")
    values = pd.to_numeric(frame["value"], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise ValueError(f"{path} contains a non-finite or non-numeric metric value.")
    return dict(zip(frame["metric"].astype(str), values.astype(float)))


def validate_run_record(metrics_path: Path, record_path: Path) -> dict[str, Any]:
    """Prove that a public run record names the exact metrics file being read."""
    if not record_path.is_file():
        raise ValueError(
            f"{metrics_path} has no public run record at {record_path}. "
            "A metrics CSV without provenance cannot be published."
        )

    record = json.loads(record_path.read_text())
    expected_run = metrics_path.name.removesuffix("_metrics.csv")
    if record.get("schema_version") != 1 or record.get("run") != expected_run:
        raise ValueError(f"{record_path} does not identify run {expected_run!r} with schema 1.")

    expected_metrics_file = f"reports/{metrics_path.name}"
    if record.get("metrics_file") != expected_metrics_file:
        raise ValueError(
            f"{record_path} names {record.get('metrics_file')!r}, not {expected_metrics_file!r}."
        )

    measured_digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    if record.get("metrics_sha256") != measured_digest:
        raise ValueError(
            f"{record_path} metric digest does not match {metrics_path}. "
            "Regenerate the run record from the checkpoint and metrics CSV."
        )

    metrics = read_metrics_csv(metrics_path)
    record_metrics = record.get("metrics")
    if not isinstance(record_metrics, dict) or record_metrics != metrics:
        raise ValueError(f"{record_path} metric values do not match {metrics_path}.")

    for key in (
        "git_sha",
        "source_tree_sha256",
        "worktree_dirty_at_training",
        "seed",
        "dataset",
        "dataset_summary",
        "source_integrity",
        "split",
    ):
        if key not in record:
            raise ValueError(f"{record_path} is missing required provenance key {key!r}.")
    source_tree_digest = record["source_tree_sha256"]
    if not isinstance(source_tree_digest, str) or len(source_tree_digest) != 64:
        raise ValueError(f"{record_path} contains an invalid training-source SHA-256.")
    source_digest = record["source_integrity"].get("expected_sha256")
    if source_digest is not None and (
        not isinstance(source_digest, str) or len(source_digest) != 64
    ):
        raise ValueError(f"{record_path} contains an invalid source SHA-256.")
    return record


def write_run_record(metadata: dict[str, Any], metrics_path: Path, destination: Path) -> Path:
    """Cross-check a checkpoint against its CSV, then write public provenance."""
    metrics = read_metrics_csv(metrics_path)
    metadata_metrics = {
        key: float(value)
        for key, value in metadata.get("metrics", {}).items()
        if isinstance(value, (int, float))
    }
    if metrics != metadata_metrics:
        missing = sorted(set(metadata_metrics) - set(metrics))
        extra = sorted(set(metrics) - set(metadata_metrics))
        mismatched = sorted(
            key
            for key in set(metrics) & set(metadata_metrics)
            if metrics[key] != metadata_metrics[key]
        )
        raise ValueError(
            f"{metrics_path} does not match checkpoint metadata. "
            f"Missing={missing}, extra={extra}, mismatched={mismatched}."
        )

    config_file = str(metadata.get("config_file", ""))
    run_name = Path(config_file).stem if config_file else str(metadata["problem"])
    public_metadata = {key: metadata[key] for key in _PUBLIC_METADATA_KEYS if key in metadata}
    if "hardware" in public_metadata:
        public_metadata["hardware"] = {
            key: public_metadata["hardware"][key]
            for key in ("platform", "machine")
            if key in public_metadata["hardware"]
        }

    record = {
        "schema_version": 1,
        "run": run_name,
        "metrics_file": f"reports/{metrics_path.name}",
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        **public_metadata,
        "metrics": metrics,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return destination
