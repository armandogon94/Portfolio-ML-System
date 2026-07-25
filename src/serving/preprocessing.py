"""Request payload -> model-ready feature frame.

**This module imports the exact feature functions training used.** It does not
reimplement them, and there is a test that fails if it ever starts to
(``tests/serving/test_skew.py``). Training/serving skew is the single most common
production ML bug and re-typing the transformations is how you get it.

The frame that reaches the model is built in three steps:

1. The request dict becomes a one-row DataFrame in the *canonical adapter schema*,
   with every column the adapter would have produced. Missing keys become NaN
   rather than 0 — LightGBM treats NaN as "unknown" and 0 as "zero", and those are
   very different statements about a transaction amount.
2. The problem's ``engineer_features`` runs with ``fit=False`` and the frequency
   maps saved at training time.
3. The result is reindexed to the checkpoint's exact ``feature_columns``, in order.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import pandas as pd

from src.config import load_config
from src.data.adapters import get_adapter
from src.features.schema import apply_category_dtypes
from src.serving.registry import LoadedModel

logger = logging.getLogger(__name__)

#: Cached per problem — reading and validating YAML on every request is wasteful.
_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def problem_config(problem: str) -> dict[str, Any]:
    """Return the (cached) config for a problem."""
    if problem not in _CONFIG_CACHE:
        _CONFIG_CACHE[problem] = load_config(problem)
    return _CONFIG_CACHE[problem]


def canonical_frame(problem: str, payload: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row frame in the adapter's canonical schema from a request.

    Args:
        problem: ``"fraud"``, ``"credit_risk"`` or ``"churn"``.
        payload: The validated request body as a dict.

    Returns:
        A single-row DataFrame carrying every canonical column. Columns the caller
        did not supply are NaN.
    """
    config = problem_config(problem)
    adapter = get_adapter(config["data"]["source"]["adapter"])

    row: dict[str, Any] = dict.fromkeys(adapter.CANONICAL_COLUMNS, float("nan"))
    unknown = set(payload) - set(adapter.CANONICAL_COLUMNS)
    if unknown:
        # Not an error: the API schemas are deliberately human-friendly subsets and
        # may carry convenience fields. Logged so a typo is findable.
        logger.debug("%s: ignoring non-canonical payload keys %s", problem, sorted(unknown))
    row.update({k: v for k, v in payload.items() if k in adapter.CANONICAL_COLUMNS})

    frame = pd.DataFrame([row])
    # The target column exists in the canonical schema but must never be populated
    # at serving time. Dropping it is cheaper than trusting the denylist here.
    frame = frame.drop(columns=[config["data"]["target"]], errors="ignore")
    return frame


def build_features(loaded: LoadedModel, payload: dict[str, Any]) -> pd.DataFrame:
    """Turn a request payload into the exact matrix the checkpoint expects.

    Args:
        loaded: The model bundle from :class:`~src.serving.registry.CheckpointRegistry`.
        payload: The validated request body.

    Returns:
        A one-row DataFrame whose columns are ``loaded.feature_columns``, in order.
    """
    config_name = str(loaded.metadata.get("config_name", loaded.problem))
    config = problem_config(config_name)
    features = importlib.import_module(config["features"]["module"])

    frame = canonical_frame(config_name, payload)
    engineered, _ = features.engineer_features(frame, loaded.feature_artifacts, fit=False)
    matrix = engineered.reindex(columns=loaded.feature_columns)

    if loaded.metadata.get("model_type") == "autoencoder":
        if loaded.preprocessor is None:
            raise ValueError(
                f"Autoencoder checkpoint {loaded.problem!r} has no saved preprocessor. "
                "Retrain it with src/training/autoencoder_pipeline.py."
            )
        transformed = loaded.preprocessor.transform(matrix)
        return pd.DataFrame(transformed, columns=loaded.feature_columns, index=matrix.index)

    for column in matrix.columns:
        if str(matrix[column].dtype) == "object":
            matrix[column] = matrix[column].astype("category")

    # Replay the exact training category sets. Without this LightGBM either
    # refuses to predict or — worse — predicts on permuted category codes.
    return apply_category_dtypes(matrix, loaded.category_dtypes)


def score(loaded: LoadedModel, payload: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    """Return ``(positive_class_probability, feature_frame)`` for one request.

    The feature frame is returned alongside the score so the explainability path
    can reuse it instead of recomputing — which would also risk the two disagreeing.
    """
    matrix = build_features(loaded, payload)
    probability = float(loaded.model.predict_proba(matrix)[0][1])
    return probability, matrix
