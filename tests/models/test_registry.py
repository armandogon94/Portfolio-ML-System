"""The model registry constructs every registered name and refuses unknown ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.registry import REGISTRY, create_model, register


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_name_constructs(name):
    model = create_model(name, {}, seed=42)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_unknown_name_raises_keyerror_listing_the_valid_names():
    """Never return None: a None model fails 40 lines later with AttributeError."""
    with pytest.raises(KeyError) as excinfo:
        create_model("randomforest")
    message = str(excinfo.value)
    assert "Unknown model" in message
    for name in REGISTRY:
        assert name in message


def test_double_registration_is_refused():
    """Silent shadowing of a model factory is a debugging nightmare."""
    with pytest.raises(ValueError, match="already registered"):

        @register("lightgbm")
        def _shadow(params, *, seed):  # pragma: no cover
            return None


def test_lightgbm_drops_early_stopping_from_the_constructor():
    """LightGBM 4.x ignores the constructor kwarg — you get 2000 trees, silently."""
    model = create_model("lightgbm", {"n_estimators": 5, "early_stopping_rounds": 10})
    assert "early_stopping_rounds" not in model.get_params()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_model_survives_nan_and_unscaled_columns(name):
    """These frames contain NaN by design and span six orders of magnitude.

    A bare LogisticRegression cannot fit them; the registry wraps it in an
    impute + scale pipeline precisely so the baseline row is real, not a crash.
    """
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        {
            "small": rng.normal(0, 1, 80),
            "huge": rng.normal(1e6, 1e5, 80),
            "sparse": np.where(rng.random(80) < 0.4, np.nan, rng.normal(0, 1, 80)),
        }
    )
    labels = rng.binomial(1, 0.3, 80)

    model = create_model(name, {"n_estimators": 5} if name in {"lightgbm", "xgboost"} else {})
    model.fit(matrix, labels)
    proba = model.predict_proba(matrix)
    assert proba.shape == (80, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_prior_baseline_is_exactly_uninformative():
    """The prior baseline must give ROC-AUC 0.5 — that is what makes it a floor."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    matrix = pd.DataFrame({"x": rng.normal(size=200)})
    labels = rng.binomial(1, 0.2, 200)

    model = create_model("prior")
    model.fit(matrix, labels)
    scores = model.predict_proba(matrix)[:, 1]

    assert len(np.unique(scores)) == 1, "a prior baseline emits one constant score"
    assert roc_auc_score(labels, scores) == pytest.approx(0.5)
