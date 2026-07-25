"""Explainability dispatch for the serving path.

Tree models (LightGBM, XGBoost) get exact SHAP via ``TreeExplainer``. The
autoencoder gets input-gradient attribution, because SHAP's ``TreeExplainer`` does
not apply and model-agnostic ``KernelExplainer`` is unsuitable for this request path.

Both explainers already existed and work; this module is only the dispatch, split
out of the old monolith so ``api.py`` holds routes and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.serving.registry import LoadedModel

logger = logging.getLogger(__name__)

#: Model types that ``shap.TreeExplainer`` handles exactly.
TREE_MODELS = {"lightgbm", "xgboost"}


def explain(loaded: LoadedModel, matrix: pd.DataFrame, *, top_n: int = 10) -> dict[str, Any]:
    """Explain one prediction.

    Args:
        loaded: The model bundle.
        matrix: The single-row feature frame produced by
            :func:`src.serving.preprocessing.build_features`. Passing the frame in
            rather than recomputing it guarantees the explanation describes the
            score that was actually returned.
        top_n: How many features to include in ``top_features``.

    Returns:
        ``{"feature_importances": {...}, "top_features": [...], "explanation_type": ...}``.

    Raises:
        NotImplementedError: The checkpoint's model type has no explainer wired up.
            Explicit is better than returning an empty explanation that a UI will
            render as "no important features".
    """
    model_type = loaded.metadata.get("model_type", "")

    if model_type in TREE_MODELS:
        from src.explainability.shap_explainer import SHAPExplainer

        return SHAPExplainer().explain(loaded.model, matrix, list(matrix.columns), top_n=top_n)

    if model_type == "autoencoder":
        return _explain_autoencoder(loaded, matrix, top_n=top_n)

    raise NotImplementedError(
        f"No explainer for model_type {model_type!r} (problem {loaded.problem!r}). "
        f"Wire one up in src/serving/explain.py rather than returning an empty "
        f"explanation — a UI cannot tell those apart."
    )


def _explain_autoencoder(
    loaded: LoadedModel, matrix: pd.DataFrame, *, top_n: int
) -> dict[str, Any]:
    """Gradient attribution over the autoencoder's reconstruction error.

    Runs on CPU deliberately. torch 2.13.0 on this hardware deadlocked a CPU
    tensor loop that followed an MPS matmul in the same process; a request handler
    is exactly the place that mixing would happen, so serving never touches MPS.
    """
    import torch

    from src.explainability.gradient_explainer import GradientExplainer

    numeric = matrix.select_dtypes(include=["number"]).fillna(0.0)
    tensor = torch.tensor(numeric.to_numpy(dtype="float32"))
    return GradientExplainer().explain(loaded.model, tensor, list(numeric.columns))
