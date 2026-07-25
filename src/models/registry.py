"""Model registry. Adding a model touches exactly one file.

``configs/<problem>.yaml`` names a model by string (``model.type: lightgbm``) and
``create_model`` turns that string into an estimator. An unknown name raises a
``KeyError`` that lists the valid names — it never returns ``None``, because a
``None`` model fails 40 lines later with a meaningless ``AttributeError``.

Every registered estimator satisfies the same three-method contract used by
``src/training/tabular.py``:

    fit(X, y, eval_set=None)  ->  self
    predict_proba(X)          ->  ndarray of shape (n, 2)
    feature_names_in_         ->  list[str]

Hardware note: LightGBM and XGBoost ship **CPU-only wheels on macOS arm64**. There
is no Metal backend for either, so ``n_jobs`` in the config is the only lever that
matters on this machine. The autoencoder is the one model here that uses MPS, and
it lives in ``src/models/autoencoder.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: name -> factory. Populated by the ``@register`` decorator at import time.
REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a model factory under ``name``.

    Raises:
        ValueError: The name is already taken. Silent shadowing of a model
            factory is a debugging nightmare, so it is refused.
    """

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if name in REGISTRY:
            raise ValueError(f"Model {name!r} is already registered by {REGISTRY[name]!r}.")
        REGISTRY[name] = factory
        return factory

    return decorator


def create_model(name: str, params: dict[str, Any] | None = None, *, seed: int = 42) -> Any:
    """Construct a registered model.

    Args:
        name: A key of :data:`REGISTRY`, e.g. ``"lightgbm"``.
        params: Hyperparameters from ``configs/<problem>.yaml`` ``model.params``.
        seed: Threaded into whichever random-state kwarg the library uses.

    Returns:
        An unfitted estimator.

    Raises:
        KeyError: ``name`` is not registered. The message lists what is.
    """
    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}") from exc
    return factory(dict(params or {}), seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────


@register("lightgbm")
def _lightgbm(params: dict[str, Any], *, seed: int):
    """LightGBM binary classifier. Handles NaN and pandas ``category`` natively.

    ``early_stopping_rounds`` is popped here and consumed by the trainer as a
    callback: passing it to the constructor is deprecated in LightGBM 4.x and
    silently ignored, which is how you end up training 2000 trees you asked to
    stop at 300.
    """
    import lightgbm as lgb

    params = dict(params)
    params.pop("early_stopping_rounds", None)
    params.setdefault("verbose", -1)
    return lgb.LGBMClassifier(random_state=seed, **params)


@register("xgboost")
def _xgboost(params: dict[str, Any], *, seed: int):
    """XGBoost binary classifier. Kept for the with/without-leakage comparison."""
    import xgboost as xgb

    params = dict(params)
    params.setdefault("eval_metric", "aucpr")
    params.setdefault("verbosity", 0)
    params.setdefault("enable_categorical", True)
    params.setdefault("tree_method", "hist")
    return xgb.XGBClassifier(random_state=seed, **params)


@register("logreg")
def _logreg(params: dict[str, Any], *, seed: int):
    """Logistic regression baseline, wrapped in the preprocessing it cannot live without.

    A bare ``LogisticRegression`` cannot consume this repo's frames at all. They
    contain NaN by design (IEEE-CIS is heavily missing), unscaled columns spanning
    six orders of magnitude, and pandas ``category`` columns. Handing it the raw
    frame raises ``could not convert string to float`` — so the "baseline" would be
    a crash, and a leaderboard with a crashed baseline row is worse than one with
    no baseline at all.

    Numeric columns: median impute then standardise.
    Categorical columns: most-frequent impute then one-hot, capped at
    ``max_categories=20``. The cap matters — ``card1`` has ~17,000 levels and an
    uncapped one-hot would produce a matrix wider than the dataset is tall.
    Unseen levels at predict time are ignored rather than raising.
    """
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    params = dict(params)
    params.setdefault("max_iter", 1000)

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=20,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [
                        ("num", numeric, make_column_selector(dtype_include="number")),
                        (
                            "cat",
                            categorical,
                            make_column_selector(dtype_include=["object", "category"]),
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(random_state=seed, **params)),
        ]
    )


@register("prior")
def _prior(params: dict[str, Any], *, seed: int):
    """Majority-class / prior-probability baseline.

    No leaderboard without a baseline row. This one predicts the training base
    rate for every row, which gives PR-AUC == base rate and ROC-AUC == 0.5
    exactly. Any model that does not beat it is not a model.
    """
    from sklearn.dummy import DummyClassifier

    del params
    return DummyClassifier(strategy="prior", random_state=seed)
