"""FastAPI inference server. Routes only.

Eight routes total: three ``/predict/*``, three ``/explain/*``, ``/models`` and
``/health``. The previous version exposed 21 routes for 4 usable checkpoints; six
of them raised 500 on every call.

Everything with logic behind it lives elsewhere: ``registry.py`` (discovery and
loading), ``preprocessing.py`` (payload -> features), ``predictors/`` (probability
-> decision), ``explain.py`` (SHAP dispatch), ``handlers.py`` (status policy),
``middleware.py`` (access log).

Port: ``8070`` inside the container is ``8000``; the compose file maps
``${BACKEND_PORT:-8070}:8000``. Never bind 8000 on the host; see
``docs/ports.example.md``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI

from src.logging_config import setup_logging
from src.serving import handlers
from src.serving.middleware import install_request_logging
from src.serving.predictors import PREDICTORS
from src.serving.registry import CheckpointRegistry
from src.serving.schemas import CardholderRequest, LoanApplicationRequest, TransactionRequest

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fintech ML Inference API",
    version="2.0.0",
    description=(
        "Fraud, credit-risk and attrition scoring with SHAP explanations. "
        "Models are discovered from checkpoints/*/metadata.json at request time."
    ),
)
registry = CheckpointRegistry()
install_request_logging(app)


# ── predictions ──────────────────────────────────────────────────────────────


@app.post("/predict/fraud", tags=["predict"])
async def predict_fraud(payload: TransactionRequest) -> dict[str, Any]:
    """Score a payment for fraud. Returns a probability and a review-queue band."""
    return handlers.predict(registry, "fraud", payload.model_dump())


@app.post("/predict/credit-risk", tags=["predict"])
async def predict_credit_risk(payload: LoanApplicationRequest) -> dict[str, Any]:
    """Score a loan application. Returns P(default) and an underwriting decision."""
    return handlers.predict(registry, "credit_risk", payload.model_dump())


@app.post("/predict/churn", tags=["predict"])
async def predict_churn(payload: CardholderRequest) -> dict[str, Any]:
    """Score a cardholder for attrition. Returns P(churn) and a retention action."""
    return handlers.predict(registry, "churn", payload.model_dump())


# ── explanations ─────────────────────────────────────────────────────────────


@app.post("/explain/fraud", tags=["explain"])
async def explain_fraud(payload: TransactionRequest) -> dict[str, Any]:
    """Per-feature SHAP contributions for a fraud score."""
    return handlers.explain(registry, "fraud", payload.model_dump())


@app.post("/explain/credit-risk", tags=["explain"])
async def explain_credit_risk(payload: LoanApplicationRequest) -> dict[str, Any]:
    """Per-feature SHAP contributions for a credit decision."""
    return handlers.explain(registry, "credit_risk", payload.model_dump())


@app.post("/explain/churn", tags=["explain"])
async def explain_churn(payload: CardholderRequest) -> dict[str, Any]:
    """Per-feature SHAP contributions for an attrition score."""
    return handlers.explain(registry, "churn", payload.model_dump())


# ── introspection ────────────────────────────────────────────────────────────


@app.get("/models", tags=["meta"])
async def models() -> dict[str, Any]:
    """Every checkpoint's metadata: metrics, git SHA, dataset and split.

    Built by globbing ``checkpoints/*/metadata.json``, so a newly trained model
    appears here with no code change and no redeploy.
    """
    return registry.model_info()


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Liveness plus per-problem availability.

    ``status`` is ``ok`` even with zero models loaded: the API is up and correctly
    reporting that nothing has been trained. Conflating "no checkpoints" with
    "unhealthy" would make the container restart-loop on a fresh clone.
    """
    info = registry.model_info()
    return {
        "status": "ok",
        "checkpoints_found": len(info),
        "models": {
            problem: {
                "available": problem in info,
                "model_type": info.get(problem, {}).get("model_type"),
                "git_sha": str(info.get(problem, {}).get("git_sha", ""))[:8] or None,
            }
            for problem in PREDICTORS
        },
    }
