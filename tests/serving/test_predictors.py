"""Predictor banding: probability -> decision, including the boundaries."""

from __future__ import annotations

import pytest

from src.serving.predictors import PREDICTORS, churn, credit_risk, fraud
from src.serving.preprocessing import build_features


def test_all_three_problems_have_a_predictor():
    assert set(PREDICTORS) == {"fraud", "credit_risk", "churn"}


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "LOW"),
        (0.19, "LOW"),
        (0.20, "MEDIUM"),
        (0.49, "MEDIUM"),
        (0.50, "HIGH"),
        (0.79, "HIGH"),
        (0.80, "CRITICAL"),
        (1.0, "CRITICAL"),
    ],
)
def test_fraud_bands_including_the_boundaries(probability, expected):
    assert fraud.risk_band(probability) == expected


def test_every_fraud_band_maps_to_an_action():
    for _, band in fraud.BANDS:
        assert band in fraud.ACTIONS


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "APPROVE"),
        (0.09, "APPROVE"),
        (0.10, "REVIEW"),
        (0.24, "REVIEW"),
        (0.25, "DECLINE"),
        (1.0, "DECLINE"),
    ],
)
def test_credit_decisions_including_the_boundaries(probability, expected):
    assert credit_risk.decision(probability) == expected


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "NO_ACTION"),
        (0.29, "NO_ACTION"),
        (0.30, "PROACTIVE_CHECKIN"),
        (0.59, "PROACTIVE_CHECKIN"),
        (0.60, "URGENT_OUTREACH"),
        (1.0, "URGENT_OUTREACH"),
    ],
)
def test_churn_actions_including_the_boundaries(probability, expected):
    assert churn.retention_action(probability) == expected


def test_churn_response_carries_the_small_n_caveat(registry, trained_churn):
    """A high score on 10,127 easy rows must not read as a strong result."""
    _, problem = trained_churn
    result = churn.predict(registry.load(problem), {"Customer_Age": 45.0})
    assert "10,127" in result["caveat"]


def test_predict_returns_the_model_version(registry, trained_churn):
    """A score with no provenance is unreviewable."""
    _, problem = trained_churn
    result = churn.predict(registry.load(problem), {"Customer_Age": 45.0})
    assert len(result["model_version"]) == 8
    assert 0.0 <= result["attrition_probability"] <= 1.0


# ── the constant-predictor guard ─────────────────────────────────────────────
# The old suite asserted only key presence and 0 <= score <= 1, which a predictor
# hardcoded to 0.5 passes. tests/test_quality_gates.py catches that — but it
# needs a real checkpoint and SKIPS on a fresh clone, so on a machine with no
# Kaggle credentials nothing was catching it at all.
#
# These two tests close that gap. They run on the fixture-built checkpoint and so
# execute everywhere, including CI.


def test_the_predictor_is_sensitive_to_its_input(registry, trained_churn):
    """Different inputs must produce different scores.

    This is the weakest true statement that a constant predictor violates. It
    cannot assert a *direction* — the CI fixture's label is independent noise, so
    there is no real relationship to be monotonic about — but "the output varies
    with the input" holds for any working model and fails for any constant one.
    """
    _, problem = trained_churn
    loaded = registry.load(problem)

    payloads = [
        {"Customer_Age": 26.0, "Months_Inactive_12_mon": 0.0, "Total_Trans_Ct": 130.0},
        {"Customer_Age": 45.0, "Months_Inactive_12_mon": 3.0, "Total_Trans_Ct": 60.0},
        {"Customer_Age": 68.0, "Months_Inactive_12_mon": 6.0, "Total_Trans_Ct": 12.0},
        {"Customer_Age": 33.0, "Months_Inactive_12_mon": 1.0, "Total_Trans_Ct": 95.0},
    ]
    scores = [churn.predict(loaded, payload)["attrition_probability"] for payload in payloads]

    assert len(set(scores)) > 1, (
        f"every input scored {scores[0]} — the predictor is ignoring its input. "
        f"A constant scorer passes any test that only checks 0 <= p <= 1."
    )


def test_the_score_comes_from_the_model_not_from_a_literal(registry, trained_churn):
    """The returned probability must equal the model's own output on that frame.

    Asserts the serving path is a pass-through of `model.predict_proba`, so a
    hardcoded return value in preprocessing.score() is caught immediately rather
    than at whatever point someone notices the dashboard is flat.
    """
    _, problem = trained_churn
    loaded = registry.load(problem)
    payload = {"Customer_Age": 45.0, "Credit_Limit": 12000.0, "Total_Trans_Ct": 67.0}

    returned = churn.predict(loaded, payload)["attrition_probability"]
    expected = float(loaded.model.predict_proba(build_features(loaded, payload))[0][1])

    assert returned == pytest.approx(expected), (
        f"serving returned {returned} but the model computes {expected} on the "
        f"same feature frame — something between them is substituting a value."
    )
