"""Publication checks for claims that must remain tied to generated evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate
from scripts.check_publication import validate_summary_documents
from src.data.adapters import credit_card_churn, lending_club

REPO_ROOT = Path(__file__).resolve().parent.parent
MEASURED_RUNS = ("fraud_ulb", "credit_risk", "churn")


def test_measured_file_backed_datasets_pin_their_sha256():
    for provenance in (lending_club.PROVENANCE, credit_card_churn.PROVENANCE):
        digest = provenance.get("expected_sha256")
        assert isinstance(digest, str) and len(digest) == 64, (
            f"{provenance['name']} has a published run but no enforced SHA-256"
        )


def test_every_measured_metrics_file_has_a_committed_run_record():
    missing = [
        run for run in MEASURED_RUNS if not (REPO_ROOT / "reports" / f"{run}_run.json").is_file()
    ]
    assert not missing, f"measured runs without committed provenance records: {missing}"


def test_published_figures_have_a_generated_evidence_manifest():
    manifest_path = REPO_ROOT / "reports" / "figures" / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generator"] == "scripts/make_figures.py --published-only"
    for filename, evidence in manifest["outputs"].items():
        figure = manifest_path.parent / filename
        assert figure.is_file()
        assert hashlib.sha256(figure.read_bytes()).hexdigest() == evidence["sha256"]


def test_evaluate_rejects_duplicate_metric_rows(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        {
            "metric": ["test_pr_auc", "test_pr_auc"],
            "value": [0.4, 0.9],
        }
    ).to_csv(reports / "credit_risk_metrics.csv", index=False)
    monkeypatch.setattr(evaluate, "get_project_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="duplicate"):
        evaluate._read("credit_risk")


def test_evaluate_rejects_a_metrics_file_that_does_not_match_its_run_record(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame({"metric": ["test_pr_auc"], "value": [0.4]}).to_csv(
        reports / "credit_risk_metrics.csv", index=False
    )
    (reports / "credit_risk_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run": "credit_risk",
                "metrics_file": "reports/credit_risk_metrics.csv",
                "metrics_sha256": "0" * 64,
                "metrics": {"test_pr_auc": 0.4},
            }
        )
    )
    monkeypatch.setattr(evaluate, "get_project_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="digest"):
        evaluate._read("credit_risk")


@pytest.mark.parametrize(
    "document",
    [REPO_ROOT / "README.md", REPO_ROOT / "reports" / "RESULTS.md"],
)
def test_every_published_figure_caption_states_what_it_does_not_support(document):
    text = document.read_text()
    starts = [index for index in range(len(text)) if text.startswith("<img ", index)]
    assert starts, f"{document} has no published figure"
    for start in starts:
        next_image = text.find("<img ", start + 1)
        caption = text[start : next_image if next_image >= 0 else len(text)]
        assert "does not support" in caption.casefold(), (
            f"figure caption after offset {start} in {document} has no limitation clause"
        )


def test_readme_does_not_publish_a_hand_maintained_coverage_result():
    text = (REPO_ROOT / "README.md").read_text()
    assert "coverage-88" not in text
    assert "88.24%" not in text


def test_publication_checker_rejects_a_mutated_summary_cell():
    readme = (REPO_ROOT / "README.md").read_text()
    results = (REPO_ROOT / "reports" / "RESULTS.md").read_text()
    original_row = next(line for line in readme.splitlines() if line.startswith("| Credit risk |"))
    mutated_row = original_row.replace("0.3935", "0.9935", 1)
    mutated_readme = readme.replace(original_row, mutated_row)
    with pytest.raises(ValueError, match="differ"):
        validate_summary_documents(mutated_readme, results, REPO_ROOT)
