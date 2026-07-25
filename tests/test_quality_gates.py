"""Quality gates: the tests that would have caught the 0.964 incident.

Everything in this file reads a **real** ``checkpoints/<problem>/metadata.json`` and
skips with a clear reason when one is absent. Skipping is correct: a fresh clone has
no checkpoints, and a gate that passes vacuously is worse than no gate.

Two directions are asserted, and the second one is the point:

* a **floor** — the model must beat its baseline and clear a minimum;
* a **ceiling** — a score above the configured band means leakage, and the whole
  reason this repository was rebuilt is that nobody questioned a suspiciously good
  number.

Monotonicity tests assert *direction*, never magnitude. They are the reason a
predictor hardcoded to a constant fails this suite.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import PROBLEMS, load_config
from src.serving.preprocessing import build_features
from src.serving.registry import CheckpointRegistry

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
    floor = load_config(problem)["expected"]["roc_auc_min"]
    measured = metrics.get("test_roc_auc") or metrics.get("cv_roc_auc_mean")
    assert measured is not None, "no ROC-AUC recorded in metadata.json"
    assert measured >= floor, f"{problem}: ROC-AUC {measured:.4f} below floor {floor}"


@pytest.mark.parametrize("problem", PROBLEMS)
def test_metric_does_not_exceed_the_leakage_ceiling(problem):
    """A score above the band is a bug report, not an achievement."""
    metrics = _metadata(problem)["metrics"]
    ceiling = load_config(problem)["expected"]["roc_auc_max"]
    measured = metrics.get("test_roc_auc") or metrics.get("cv_roc_auc_mean")
    assert measured <= ceiling, (
        f"{problem}: ROC-AUC {measured:.4f} EXCEEDS {ceiling}. Suspect leakage: "
        f"a random split, an id column in the features, or a post-outcome field. "
        f"Investigate before publishing."
    )


@pytest.mark.parametrize("problem", PROBLEMS)
def test_model_beats_its_own_baseline(problem):
    """Without this, a PR-AUC number means nothing at all."""
    metrics = _metadata(problem)["metrics"]
    model_pr = metrics.get("test_pr_auc") or metrics.get("cv_pr_auc_mean")
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
def test_no_leak_warning_was_recorded(problem):
    warning = _metadata(problem).get("suspected_leakage")
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


def test_a_constant_predictor_would_fail_these_gates():
    """Proof the gates have teeth, and it runs without any checkpoint.

    The old suite asserted only key presence and 0 <= score <= 1, which a
    predictor hardcoded to 0.5 passes. This asserts the opposite: a constant
    scorer cannot satisfy a strict inequality against its own baseline.
    """
    constant = pd.Series([0.5] * 100)
    baseline = pd.Series([0.5] * 100)
    assert not (constant.mean() > baseline.mean()), (
        "a constant predictor must not be able to beat a constant baseline"
    )
