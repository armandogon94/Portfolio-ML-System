"""One module per problem: the business framing that turns a probability into a decision.

Each module exposes ``predict(loaded, payload) -> dict``. The probability comes
from ``src/serving/preprocessing.score``; what each module adds is the threshold
policy — the part a domain expert would argue about and an ML engineer should not
bury in a route handler.

Thresholds here are **documented conventions, not calibrated policy.** Calibrating
them needs the cost of a false positive versus a false negative, which is a
business input this repository does not have. Each module says so.
"""

from __future__ import annotations

from src.serving.predictors import churn, credit_risk, fraud

#: problem name -> predict callable.
PREDICTORS = {
    "fraud": fraud.predict,
    "credit_risk": credit_risk.predict,
    "churn": churn.predict,
}

__all__ = ["PREDICTORS", "churn", "credit_risk", "fraud"]
