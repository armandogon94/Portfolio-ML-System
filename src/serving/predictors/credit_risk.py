"""Credit decisioning: default probability -> approve / review / decline.

Deliberately blunt, and deliberately annotated as such. A real lender sets these
cut points from an expected-loss calculation: approve while
``P(default) x loss_given_default < interest_income``. That needs a pricing model
this repository does not contain, so the thresholds below are illustrative.

``reports/RESULTS.md`` argues the same point at length: on consumer credit,
**expected loss at a chosen approval rate is a better decision metric than AUC**,
and a model with ROC-AUC 0.70 can be entirely usable.
"""

from __future__ import annotations

from typing import Any

from src.serving.preprocessing import score
from src.serving.registry import LoadedModel

#: (upper bound on P(default), decision).
BANDS = (
    (0.10, "APPROVE"),
    (0.25, "REVIEW"),
    (1.01, "DECLINE"),
)


def decision(probability: float) -> str:
    """Map a default probability to an underwriting decision."""
    for upper, name in BANDS:
        if probability < upper:
            return name
    return "DECLINE"


def predict(loaded: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    """Score a loan application.

    Args:
        loaded: The credit-risk checkpoint bundle.
        payload: Validated request body.

    Returns:
        ``default_probability``, ``decision``, ``model_version``, ``trained_on``
        and the ``feature_frame``.
    """
    probability, matrix = score(loaded, payload)
    return {
        "default_probability": probability,
        "decision": decision(probability),
        # Stated explicitly so nobody mistakes a demo threshold for a credit policy.
        "threshold_basis": (
            "Illustrative cut points, not a calibrated credit policy. A real "
            "decision needs expected loss at a target approval rate."
        ),
        "model_version": str(loaded.metadata.get("git_sha", "unknown"))[:8],
        "trained_on": loaded.metadata.get("dataset", {}).get("slug", "unknown"),
        "feature_frame": matrix,
    }
