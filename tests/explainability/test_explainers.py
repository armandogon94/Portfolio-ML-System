"""SHAP and gradient explainers return contributions in the documented shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import SHAPExplainer


@pytest.fixture
def tree_model():
    import lightgbm as lgb

    rng = np.random.default_rng(11)
    matrix = pd.DataFrame(
        {"a": rng.normal(size=200), "b": rng.normal(size=200), "c": rng.normal(size=200)}
    )
    # 'a' carries all the signal, so SHAP must rank it first.
    labels = (matrix["a"] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=42)
    model.fit(matrix, labels)
    return model, matrix


def test_shap_returns_the_documented_shape(tree_model):
    model, matrix = tree_model
    result = SHAPExplainer().explain(model, matrix.head(1), list(matrix.columns))

    assert result["explanation_type"] == "shap"
    assert set(result["feature_importances"]) == set(matrix.columns)
    assert isinstance(result["top_features"], list)
    assert set(result["top_features"][0]) == {"feature", "importance"}


def test_shap_ranks_the_informative_feature_first(tree_model):
    """An explainer that cannot find a feature the label is a function of is broken."""
    model, matrix = tree_model
    result = SHAPExplainer().explain(model, matrix.head(1), list(matrix.columns))
    assert result["top_features"][0]["feature"] == "a"


def test_top_features_are_sorted_by_absolute_contribution(tree_model):
    model, matrix = tree_model
    result = SHAPExplainer().explain(model, matrix.head(1), list(matrix.columns))
    magnitudes = [abs(entry["importance"]) for entry in result["top_features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_n_is_respected(tree_model):
    model, matrix = tree_model
    result = SHAPExplainer().explain(model, matrix.head(1), list(matrix.columns), top_n=2)
    assert len(result["top_features"]) == 2


def test_gradient_explainer_attributes_over_reconstruction_error():
    import torch

    from src.explainability.gradient_explainer import GradientExplainer
    from src.models.autoencoder import FraudAutoencoder

    torch.manual_seed(0)
    model = FraudAutoencoder(input_dim=4, hidden_dims=[8, 4])
    result = GradientExplainer().explain(model, torch.randn(1, 4), ["w", "x", "y", "z"])

    assert set(result["feature_importances"]) == {"w", "x", "y", "z"}
    assert result["top_features"]
