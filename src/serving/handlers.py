"""The two shared request handlers, plus their HTTP status policy.

Split out of ``api.py`` so that file is route declarations and nothing else. The
non-obvious content here is the status mapping, which is a deliberate choice:

``503``: no checkpoint on disk.
    The service is healthy; the model simply has not been trained. A ``500`` would
    say "bug" and send a reviewer reading tracebacks for a fresh clone that is
    behaving exactly as documented. The detail carries the command to fix it.

``501``: the model type has no explainer wired up.
    "Not implemented" is the truth. Returning an empty explanation instead would
    render in the UI as "this prediction had no important features", which is a
    different and false statement.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.serving import explain as explain_module
from src.serving.predictors import PREDICTORS
from src.serving.preprocessing import build_features
from src.serving.registry import CheckpointRegistry


def _load_or_503(registry: CheckpointRegistry, problem: str):
    try:
        return registry.load(problem)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def predict(registry: CheckpointRegistry, problem: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Score one request and strip the internal feature frame from the response."""
    loaded = _load_or_503(registry, problem)
    result = PREDICTORS[problem](loaded, payload)
    result.pop("feature_frame", None)  # internal, not part of the response contract
    return result


def explain(registry: CheckpointRegistry, problem: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Explain one request using the same feature frame the score would use."""
    loaded = _load_or_503(registry, problem)
    try:
        matrix = build_features(loaded, payload)
        return explain_module.explain(loaded, matrix)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
