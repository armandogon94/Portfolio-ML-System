"""Attrition decisioning: churn probability -> retention action.

Same caveat as the other two predictors: the cut points are a readable convention.
A retention team would set them from the cost of an outreach contact against the
lifetime value of the retained customer.

One extra caveat specific to this problem, repeated in the README: the underlying
dataset is **10,127 rows** and well separated. A high score here reflects an easy
dataset, not a strong model. It is not the flagship result.
"""

from __future__ import annotations

from typing import Any

from src.serving.preprocessing import score
from src.serving.registry import LoadedModel

#: (upper bound on P(attrition), action).
BANDS = (
    (0.30, "NO_ACTION"),
    (0.60, "PROACTIVE_CHECKIN"),
    (1.01, "URGENT_OUTREACH"),
)


def retention_action(probability: float) -> str:
    """Map an attrition probability to a retention action."""
    for upper, name in BANDS:
        if probability < upper:
            return name
    return "URGENT_OUTREACH"


def predict(loaded: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    """Score a cardholder for attrition risk.

    Args:
        loaded: The churn checkpoint bundle.
        payload: Validated request body.

    Returns:
        ``attrition_probability``, ``retention_action``, ``model_version``,
        ``trained_on``, a ``caveat`` and the ``feature_frame``.
    """
    probability, matrix = score(loaded, payload)
    return {
        "attrition_probability": probability,
        "retention_action": retention_action(probability),
        "caveat": (
            "n = 10,127 and the dataset is easy. Metrics are 5-fold CV mean +/- std; "
            "see reports/RESULTS.md before reading much into a high score."
        ),
        "model_version": str(loaded.metadata.get("git_sha", "unknown"))[:8],
        "trained_on": loaded.metadata.get("dataset", {}).get("slug", "unknown"),
        "feature_frame": matrix,
    }
