"""Fraud decisioning: probability -> review-queue band.

The bands mirror how a payments review team actually works: almost everything is
approved automatically, a thin slice goes to a human queue, and a thinner slice is
declined outright. The cut points below are **conventions chosen for a readable
demo**, not a calibrated policy. Calibrating them weighs the cost of a
false decline (a lost customer) against the cost of a missed fraud (a chargeback),
and this repository has no such business input. ``reports/RESULTS.md`` reports
``precision_at_1pct`` precisely so a reader can pick their own cut point.
"""

from __future__ import annotations

from typing import Any

from src.serving.preprocessing import score
from src.serving.registry import LoadedModel

#: Upper bound of each band, in ascending order.
BANDS = (
    (0.20, "LOW"),
    (0.50, "MEDIUM"),
    (0.80, "HIGH"),
    (1.01, "CRITICAL"),
)

#: Action attached to each band.
ACTIONS = {
    "LOW": "AUTO_APPROVE",
    "MEDIUM": "MONITOR",
    "HIGH": "MANUAL_REVIEW",
    "CRITICAL": "DECLINE",
}


def risk_band(probability: float) -> str:
    """Map a fraud probability to a named band."""
    for upper, name in BANDS:
        if probability < upper:
            return name
    return "CRITICAL"


def predict(loaded: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    """Score a transaction and attach the review decision.

    Args:
        loaded: The fraud checkpoint bundle.
        payload: Validated request body.

    Returns:
        ``fraud_probability``, ``risk_band``, ``recommended_action``,
        ``model_version`` (the git SHA that trained the checkpoint) and the
        ``feature_frame`` for the explainability path to reuse.
    """
    probability, matrix = score(loaded, payload)
    band = risk_band(probability)
    return {
        "fraud_probability": probability,
        "risk_band": band,
        "recommended_action": ACTIONS[band],
        "model_version": str(loaded.metadata.get("git_sha", "unknown"))[:8],
        "trained_on": loaded.metadata.get("dataset", {}).get("slug", "unknown"),
        "feature_frame": matrix,
    }
