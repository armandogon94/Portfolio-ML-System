#!/usr/bin/env python
"""Validate every active result table and committed figure against generated evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import _pick
from src.config import get_project_root
from src.training.publication import validate_run_record
from src.training.tabular import _source_tree_sha256

RUNS = ("fraud_ulb", "credit_risk", "churn")
METRIC_COLUMNS = (
    "pr_auc",
    "roc_auc",
    "pr_auc_baseline",
    "pr_auc_delta",
    "precision_at_1pct",
    "recall_at_1pct_fpr",
)
README_LABELS = {
    "fraud_ulb": "Fraud (`fraud_ulb`, credential-free)",
    "credit_risk": "Credit risk",
    "churn": "Churn",
}


def _table_row(text: str, label: str) -> list[str]:
    prefix = f"| {label} |"
    matches = [line for line in text.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one table row beginning {prefix!r}, found {len(matches)}.")
    return [cell.strip().replace("**", "") for cell in matches[0].strip("|").split("|")]


def _metrics_and_records(root: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    metrics = {}
    records = {}
    current_source_digest = _source_tree_sha256()
    for run in RUNS:
        metrics_path = root / "reports" / f"{run}_metrics.csv"
        record = validate_run_record(metrics_path, root / "reports" / f"{run}_run.json")
        if record["source_tree_sha256"] != current_source_digest:
            raise ValueError(
                f"{run} was trained from a different source tree. "
                "Retrain it before publishing this code."
            )
        metrics[run] = record["metrics"]
        records[run] = record
    return metrics, records


def validate_summary_documents(readme: str, results: str, root: Path) -> None:
    """Reject any active summary cell that differs from its validated metrics CSV."""
    metrics, _ = _metrics_and_records(root)
    for run in RUNS:
        expected = [_pick(metrics[run], name) for name in METRIC_COLUMNS]

        readme_row = _table_row(readme, README_LABELS[run])
        readme_cells = readme_row[4:8]
        if readme_cells != expected[:4]:
            raise ValueError(
                f"README result cells for {run} differ from generated evidence: "
                f"expected {expected[:4]}, found {readme_cells}."
            )

        results_row = _table_row(results, f"`{run}`")
        results_cells = results_row[1:7]
        if results_cells != expected:
            raise ValueError(
                f"RESULTS summary cells for {run} differ from generated evidence: "
                f"expected {expected}, found {results_cells}."
            )


def validate_dataset_claims(readme: str, results: str, root: Path) -> None:
    """Validate active dataset counts and the complete temporal split table."""
    _, records = _metrics_and_records(root)
    expected_readme_fragments = {
        "fraud_ulb": (
            f"{records['fraud_ulb']['dataset_summary']['n_rows']:,} rows · "
            f"{records['fraud_ulb']['n_features']} features · "
            f"{records['fraud_ulb']['dataset_summary']['positive_rate']:.4%}"
        ),
        "credit_risk": (
            f"{records['credit_risk']['dataset_summary']['n_rows']:,} terminal-status rows · "
            f"{records['credit_risk']['n_features']} features · "
            f"{records['credit_risk']['dataset_summary']['positive_rate']:.2%} default"
        ),
        "churn": (
            f"{records['churn']['dataset_summary']['n_rows']:,} rows · "
            f"{records['churn']['n_features']} features · "
            f"{records['churn']['dataset_summary']['positive_rate']:.2%}"
        ),
    }
    for run, fragment in expected_readme_fragments.items():
        if fragment not in _table_row(readme, README_LABELS[run])[1]:
            raise ValueError(
                f"README dataset summary for {run} is not generated from its run record."
            )

    credit_split = json.loads((root / "reports" / "credit_risk_split.json").read_text())
    for partition in credit_split["partitions"]:
        label = {"train": "Train", "val": "Validation", "test": "Test"}[partition["partition"]]
        row = _table_row(results, label)
        expected = [
            label,
            f"{partition['n']:,}",
            f"{partition['n_positive']:,}",
            f"{partition['positive_rate']:.4f}",
            partition["min"].split()[0],
            partition["max"].split()[0],
        ]
        if row != expected:
            raise ValueError(
                f"Credit-risk split row differs from generated evidence: "
                f"expected {expected}, found {row}."
            )

    fraud_split = json.loads((root / "reports" / "fraud_ulb_split.json").read_text())
    for partition in fraud_split["partitions"]:
        label = {
            "train": "ULB train",
            "val": "ULB validation",
            "test": "ULB test",
        }[partition["partition"]]
        row = _table_row(results, label)
        expected = [
            label,
            f"{partition['n']:,}",
            f"{partition['n_positive']:,}",
            f"{partition['positive_rate']:.4f}",
            partition["min"],
            partition["max"],
        ]
        if row != expected:
            raise ValueError(
                f"ULB split row differs from generated evidence: expected {expected}, found {row}."
            )


def validate_figure_evidence(readme: str, results: str, root: Path) -> None:
    """Validate PNG bytes, source digests, and caption-only measurements."""
    manifest_path = root / "reports" / "figures" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("generator") != "scripts/make_figures.py --published-only":
        raise ValueError("The figure manifest does not name the committed generator.")

    _, records = _metrics_and_records(root)
    for filename, output in manifest["outputs"].items():
        figure_path = manifest_path.parent / filename
        measured = hashlib.sha256(figure_path.read_bytes()).hexdigest()
        if output["sha256"] != measured:
            raise ValueError(f"{figure_path} differs from the generated figure manifest.")

    for run in RUNS:
        source = manifest["sources"][run]
        if source["metrics_sha256"] != records[run]["metrics_sha256"]:
            raise ValueError(f"{run} figure metrics digest differs from its run record.")
        if source["predictions_kind"] == "out_of_fold":
            expected_rows = records[run]["dataset_summary"]["n_rows"]
            expected_positive = records[run]["dataset_summary"]["positive_count"]
        else:
            expected_rows = int(records[run]["metrics"]["test_n"])
            expected_positive = round(expected_rows * records[run]["metrics"]["test_positive_rate"])
        if source["n_rows"] != expected_rows or source["positive_count"] != expected_positive:
            raise ValueError(f"{run} figure sample counts differ from its run record.")
        observed = source["calibration_observed_rate"]
        if any(right < left for left, right in zip(observed, observed[1:])):
            raise ValueError(f"{run} calibration rates do not support the finding title.")
        pooled = source["pooled_average_precision"]
        if not pooled["model"] > pooled["logistic_regression"]:
            raise ValueError(f"{run} PR rows do not support the finding title.")

    caption_claim = (
        f"ULB fraud has a held-out Brier score of "
        f"{records['fraud_ulb']['metrics']['test_brier']:.5f} with "
        f"{manifest['sources']['fraud_ulb']['calibration_zero_event_bins']} of 10 bins "
        f"containing no observed positives; consumer credit risk has a held-out "
        f"Brier score of {records['credit_risk']['metrics']['test_brier']:.5f} with "
        f"{manifest['sources']['credit_risk']['calibration_zero_event_bins']} of 10 bins "
        f"containing no observed positives; card attrition has a mean fold Brier score of "
        f"{records['churn']['metrics']['cv_brier_mean']:.5f} with "
        f"{manifest['sources']['churn']['calibration_zero_event_bins']} of 10 bins "
        f"containing no observed positives."
    )
    for name, document in (("README", readme), ("RESULTS", results)):
        if caption_claim not in document:
            raise ValueError(f"{name} calibration caption differs from generated figure evidence.")


def main() -> int:
    root = get_project_root()
    readme = (root / "README.md").read_text()
    results = (root / "reports" / "RESULTS.md").read_text()
    validate_summary_documents(readme, results, root)
    validate_dataset_claims(readme, results, root)
    validate_figure_evidence(readme, results, root)
    print("Publication evidence is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
