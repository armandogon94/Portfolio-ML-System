"""Predictor banding: probability -> decision, including the boundaries."""

from __future__ import annotations

import pytest

from src.serving.predictors import PREDICTORS, churn, credit_risk, fraud


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
