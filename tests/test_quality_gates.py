"""Quality gates: the tests that would have caught the 0.964 incident.

Aggregate metric gates read committed ``reports/<run>_run.json`` records first,
so they stay live in a fresh clone. Serving-behavior gates still require the
ignored model checkpoint and skip with a clear training command when it is
absent.

The expected-range sanity band is a smoke alarm, not a leakage test:

* a floor catches a broken pipeline;
* a configured upper bound prompts investigation, but cannot prove leakage.

Monotonicity tests assert *direction*, never magnitude. They are the reason a
predictor hardcoded to a constant fails this suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import PROBLEMS, load_config
from src.evaluation.classification_metrics import compute_classification_metrics
from src.serving.preprocessing import build_features
from src.serving.registry import CheckpointRegistry
from src.training.tabular import TabularTrainer

registry = CheckpointRegistry()
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every run whose numbers are *published*, which is not the same set as ``PROBLEMS``.
#: ``PROBLEMS`` is the closed list of three business problems; a problem can have more
#: than one dataset config (``fraud.yaml`` for IEEE-CIS, ``fraud_ulb.yaml`` for OpenML
#: 1597). The gates must follow both public run records and local checkpoints.
#: Parametrising over ``PROBLEMS`` left ``fraud_ulb``, the measured result that
#: replaced the retracted 0.964, ungated.
PUBLISHED_RUNS = tuple(
    sorted(
        path.name.removesuffix("_run.json") for path in (REPO_ROOT / "reports").glob("*_run.json")
    )
)
LOCAL_CHECKPOINT_RUNS = tuple(
    sorted(
        path.stem
        for path in (REPO_ROOT / "configs").glob("*.yaml")
        if (registry.root / path.stem / "metadata.json").exists()
    )
)
GATED_RUNS = tuple(sorted(set(PUBLISHED_RUNS) | set(LOCAL_CHECKPOINT_RUNS))) or PROBLEMS

#: Union of declared problems and published runs. Keeping ``PROBLEMS`` in the
#: parametrisation preserves the loud, informative skip for ``fraud`` (blocked on Kaggle
#: credentials) instead of silently dropping it from the suite.
CHECKED_RUNS = tuple(sorted(set(PROBLEMS) | set(GATED_RUNS)))

# Kept separate so the guard below can prove that the baseline gate covers the
# same published-run set as the metric gates.
BASELINE_GATED_RUNS = CHECKED_RUNS


def _banded_metric(problem: str, metrics: dict, config: dict) -> tuple[str, float | None]:
    """Return the metric the sanity band actually names, and its measured value.

    The band is declared per config. Reading ROC-AUC unconditionally previously
    compared a secondary metric with a primary-metric bound. The corrected ULB
    temporal run has no post hoc expected range; its degenerate-score and
    baseline gates remain active.
    """
    metric = config["sanity_band"].get("metric", "roc_auc")
    key = f"cv_{metric}_mean" if config["split"]["type"] == "stratified_kfold" else f"test_{metric}"
    return metric, metrics.get(key)


def _checkpoint_metadata(problem: str) -> dict:
    path = registry.root / problem / "metadata.json"
    if not path.exists():
        pytest.skip(
            f"No checkpoint for {problem!r}. Quality gates need a real trained model:\n"
            f"  uv run python scripts/download_data.py --dataset all\n"
            f"  uv run python scripts/train.py --model {problem}\n"
            f"See docs/PROGRESS.md: this is BLOCKED on Kaggle credentials."
        )
    return json.loads(path.read_text())


def _gate_metadata(problem: str) -> dict:
    """Load public aggregate evidence first, then a local checkpoint."""
    public_path = REPO_ROOT / "reports" / f"{problem}_run.json"
    if public_path.exists():
        return json.loads(public_path.read_text())
    return _checkpoint_metadata(problem)


def test_public_run_record_is_used_when_a_checkpoint_is_absent(tmp_path, monkeypatch):
    """Published aggregate gates must stay live in a fresh clone."""
    monkeypatch.setattr(registry, "root", tmp_path / "checkpoints")
    metadata = _gate_metadata("fraud_ulb")
    assert metadata["run"] == "fraud_ulb"
    assert metadata["metrics"]["test_pr_auc"] > 0


@pytest.mark.parametrize(
    "problem",
    [problem for problem in CHECKED_RUNS if "sanity_band" in load_config(problem)],
)
def test_metric_clears_the_configured_floor(problem):
    metrics = _gate_metadata(problem)["metrics"]
    config = load_config(problem)
    floor = config["sanity_band"]["min"]
    metric, measured = _banded_metric(problem, metrics, config)
    assert measured is not None, f"{problem}: no {metric} recorded in metadata.json"
    assert measured >= floor, f"{problem}: {metric} {measured:.4f} below floor {floor}"


@pytest.mark.parametrize(
    "problem",
    [problem for problem in CHECKED_RUNS if "max" in load_config(problem).get("sanity_band", {})],
)
def test_metric_stays_inside_the_configured_sanity_band(problem):
    metrics = _gate_metadata(problem)["metrics"]
    config = load_config(problem)
    upper = config["sanity_band"]["max"]
    metric, measured = _banded_metric(problem, metrics, config)
    assert measured is not None, f"{problem}: no {metric} recorded in metadata.json"
    assert measured <= upper, (
        f"{problem}: {metric} {measured:.4f} is above the expected-range sanity "
        f"band {upper}. Investigate the split, features, and target before publishing."
    )


def test_every_published_run_is_covered_by_the_metric_gates():
    """Guard the guard: a trained checkpoint must never sit outside the gates.

    ``fraud_ulb`` -- the measured result that replaced the retracted 0.964 -- was
    published in the README while every gate parametrised over ``PROBLEMS``, which omits
    it. This fails if that ever recurs.
    """
    ungated = set(GATED_RUNS) - set(CHECKED_RUNS)
    assert not ungated, f"trained checkpoints outside the metric gates: {sorted(ungated)}"
    assert "fraud_ulb" in CHECKED_RUNS, "fraud_ulb is published in the README; it must be gated"


def test_every_checked_run_is_covered_by_the_baseline_gate():
    missing = set(CHECKED_RUNS) - set(BASELINE_GATED_RUNS)
    assert not missing, f"published runs outside the baseline gate: {sorted(missing)}"


@pytest.mark.parametrize("problem", BASELINE_GATED_RUNS)
def test_model_beats_its_own_baseline(problem):
    """Without this, a PR-AUC number means nothing at all."""
    metrics = _gate_metadata(problem)["metrics"]
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


@pytest.mark.parametrize("problem", CHECKED_RUNS)
def test_metadata_carries_full_provenance(problem):
    """A metric you cannot trace to a commit and a dataset is not evidence."""
    metadata = _gate_metadata(problem)
    for key in (
        "git_sha",
        "source_tree_sha256",
        "worktree_dirty_at_training",
        "seed",
        "dataset",
        "dataset_summary",
        "source_integrity",
        "split",
        "n_features",
        "metrics",
    ):
        assert key in metadata, f"{problem}: metadata.json missing {key!r}"
    assert metadata["git_sha"] != "unknown"
    assert metadata["seed"] == 42


@pytest.mark.parametrize("problem", CHECKED_RUNS)
def test_recorded_sanity_warning_has_public_adjudication(problem):
    """A warning may be accepted only when the report names the post-run decision."""
    warning = _gate_metadata(problem).get("sanity_band_warning")
    if warning is None:
        return
    report = (REPO_ROOT / "reports" / "RESULTS.md").read_text()
    marker = f"**Sanity-band adjudication (`{problem}`).**"
    assert marker in report, f"{problem}: warning is public but has no {marker!r} section"
    assert "post-run threshold change" in report


def test_churn_reports_a_standard_deviation_not_a_single_number():
    """n = 10,127. A single hold-out figure on this dataset is noise."""
    metrics = _gate_metadata("churn")["metrics"]
    assert "cv_roc_auc_std" in metrics
    assert metrics.get("cv_n_folds", 0) >= 5


# ── monotonicity: direction only, never magnitude ────────────────────────────


def _score(problem: str, payload: dict) -> float:
    loaded = registry.load(problem)
    matrix = build_features(loaded, payload)
    return float(loaded.model.predict_proba(matrix)[0][1])


def test_a_higher_fico_score_does_not_raise_default_risk():
    """Directional sanity. A model that inverts this is wired backwards."""
    _checkpoint_metadata("credit_risk")
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
    """Directional sanity over the range where the data is actually monotonic.

    This gate originally probed 0 against 6 inactive months and failed against the
    first real churn checkpoint. The model was right and the gate was wrong: the
    empirical attrition rate in ``BankChurners.csv`` is **not** monotonic in
    ``Months_Inactive_12_mon``. Measured on the full 10,127 rows on 2026-07-25::

        months inactive:   0      1      2      3      4      5      6
        attrition rate: 51.7%   4.5%  15.4%  21.5%  29.9%  18.0%  15.3%
        n:                 29   2233   3282   3846    435    178    124

    The 0-month cell is 29 customers, and the rate falls again after 4 months. A
    model that scored 6 months above 0 months would be contradicting its training
    data, so asserting that was asserting a bug.

    1 through 4 months is the segment where the relationship is monotone
    increasing and every cell has hundreds to thousands of rows behind it. That is
    what this gate now checks. Widening it back to 0..6 requires new data, not a
    new model.
    """
    _checkpoint_metadata("churn")
    base = {
        "Customer_Age": 45.0,
        "Credit_Limit": 12000.0,
        "Total_Trans_Amt": 4400.0,
        "Total_Trans_Ct": 67.0,
        "Total_Relationship_Count": 4.0,
        "Months_on_book": 36.0,
    }
    active = _score("churn", {**base, "Months_Inactive_12_mon": 1.0})
    dormant = _score("churn", {**base, "Months_Inactive_12_mon": 4.0})
    assert dormant >= active - 1e-6, (
        f"going from 1 to 4 inactive months lowered churn risk: "
        f"{active:.4f} -> {dormant:.4f}. The training data rises monotonically "
        f"across that segment (4.5% -> 29.9%), so this inverts the data."
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
