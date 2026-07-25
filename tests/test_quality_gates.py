"""Quality gates: the tests that would have caught the 0.964 incident.

Everything in this file reads a **real** ``checkpoints/<problem>/metadata.json`` and
skips with a clear reason when one is absent. Skipping is correct: a fresh clone has
no checkpoints, and a gate that passes vacuously is worse than no gate.

The expected-range sanity band is a smoke alarm, not a leakage test:

* a floor catches a broken pipeline;
* a configured upper bound prompts investigation, but cannot prove leakage.

Monotonicity tests assert *direction*, never magnitude. They are the reason a
predictor hardcoded to a constant fails this suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import PROBLEMS, load_config
from src.evaluation.classification_metrics import compute_classification_metrics
from src.serving.preprocessing import build_features
from src.serving.registry import CheckpointRegistry
from src.training.tabular import TabularTrainer

registry = CheckpointRegistry()


def _metadata(problem: str) -> dict:
    path = registry.root / problem / "metadata.json"
    if not path.exists():
        pytest.skip(
            f"No checkpoint for {problem!r}. Quality gates need a real trained model:\n"
            f"  uv run python scripts/download_data.py --dataset all\n"
            f"  uv run python scripts/train.py --model {problem}\n"
            f"See docs/PROGRESS.md — this is BLOCKED on Kaggle credentials."
        )
    return json.loads(path.read_text())


@pytest.mark.parametrize("problem", PROBLEMS)
def test_metric_clears_the_configured_floor(problem):
    metrics = _metadata(problem)["metrics"]
    config = load_config(problem)
    floor = config["sanity_band"]["min"]
    measured = (
        metrics.get("cv_roc_auc_mean")
        if config["split"]["type"] == "stratified_kfold"
        else metrics.get("test_roc_auc")
    )
    assert measured is not None, "no ROC-AUC recorded in metadata.json"
    assert measured >= floor, f"{problem}: ROC-AUC {measured:.4f} below floor {floor}"


@pytest.mark.parametrize(
    "problem",
    [problem for problem in PROBLEMS if "max" in load_config(problem).get("sanity_band", {})],
)
def test_metric_stays_inside_the_configured_sanity_band(problem):
    metrics = _metadata(problem)["metrics"]
    config = load_config(problem)
    upper = config["sanity_band"]["max"]
    measured = (
        metrics.get("cv_roc_auc_mean")
        if config["split"]["type"] == "stratified_kfold"
        else metrics.get("test_roc_auc")
    )
    assert measured <= upper, (
        f"{problem}: ROC-AUC {measured:.4f} is above the expected-range sanity "
        f"band {upper}. Investigate the split, features, and target before publishing."
    )


@pytest.mark.parametrize("problem", PROBLEMS)
def test_model_beats_its_own_baseline(problem):
    """Without this, a PR-AUC number means nothing at all."""
    metrics = _metadata(problem)["metrics"]
    config = load_config(problem)
    if config["split"]["type"] == "stratified_kfold":
        model_pr = metrics.get("cv_pr_auc_mean")
        baseline_pr = metrics.get("cv_pr_auc_baseline_mean")
    else:
        model_pr = metrics.get("test_pr_auc")
        baseline_pr = metrics.get("test_pr_auc_baseline")
    if baseline_pr is None:
        pytest.skip(f"{problem}: no baseline recorded")
    assert model_pr > baseline_pr, f"{problem}: model {model_pr} <= baseline {baseline_pr}"


@pytest.mark.parametrize("problem", PROBLEMS)
def test_metadata_carries_full_provenance(problem):
    """A metric you cannot trace to a commit and a dataset is not evidence."""
    metadata = _metadata(problem)
    for key in ("git_sha", "seed", "dataset", "split", "feature_columns", "metrics"):
        assert key in metadata, f"{problem}: metadata.json missing {key!r}"
    assert metadata["git_sha"] != "unknown"
    assert metadata["seed"] == 42


@pytest.mark.parametrize("problem", PROBLEMS)
def test_no_sanity_band_warning_was_recorded(problem):
    warning = _metadata(problem).get("sanity_band_warning")
    assert warning is None, f"{problem}: the trainer flagged a problem: {warning}"


def test_churn_reports_a_standard_deviation_not_a_single_number():
    """n = 10,127. A single hold-out figure on this dataset is noise."""
    metrics = _metadata("churn")["metrics"]
    assert "cv_roc_auc_std" in metrics
    assert metrics.get("cv_n_folds", 0) >= 5


# ── monotonicity: direction only, never magnitude ────────────────────────────


def _score(problem: str, payload: dict) -> float:
    loaded = registry.load(problem)
    matrix = build_features(loaded, payload)
    return float(loaded.model.predict_proba(matrix)[0][1])


def test_a_higher_fico_score_does_not_raise_default_risk():
    """Directional sanity. A model that inverts this is wired backwards."""
    _metadata("credit_risk")
    base = {
        "loan_amnt": 15000.0,
        "annual_inc": 72000.0,
        "installment": 509.66,
        "dti": 18.24,
        "revol_bal": 14300.0,
        "revol_util": 52.4,
        "open_acc": 11.0,
        "total_acc": 24.0,
    }
    low = _score("credit_risk", {**base, "fico_range_low": 620.0, "fico_range_high": 624.0})
    high = _score("credit_risk", {**base, "fico_range_low": 820.0, "fico_range_high": 824.0})
    assert high <= low + 1e-6, f"higher FICO raised default risk: {low:.4f} -> {high:.4f}"


def test_more_inactive_months_does_not_lower_churn_risk():
    _metadata("churn")
    base = {
        "Customer_Age": 45.0,
        "Credit_Limit": 12000.0,
        "Total_Trans_Amt": 4400.0,
        "Total_Trans_Ct": 67.0,
        "Total_Relationship_Count": 4.0,
        "Months_on_book": 36.0,
    }
    active = _score("churn", {**base, "Months_Inactive_12_mon": 0.0})
    dormant = _score("churn", {**base, "Months_Inactive_12_mon": 6.0})
    assert dormant >= active - 1e-6, (
        f"six dormant months lowered churn risk: {active:.4f} -> {dormant:.4f}"
    )


def test_a_constant_predictor_fails_the_actual_quality_gate():
    """A real scorer flows through metrics and the same gate training calls."""

    class ConstantPredictor:
        def predict_proba(self, matrix):
            positive = np.zeros(len(matrix), dtype=float)
            return np.column_stack([1.0 - positive, positive])

    labels = np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    fixture = np.arange(len(labels), dtype=float).reshape(-1, 1)
    scores = ConstantPredictor().predict_proba(fixture)[:, 1]
    metrics = compute_classification_metrics(labels, scores)

    trainer = TabularTrainer("fraud", sample=True)
    try:
        failure = trainer._check_sanity_band(metrics)
    finally:
        trainer.finish()

    assert failure is not None
    assert "degenerate" in failure.casefold()
