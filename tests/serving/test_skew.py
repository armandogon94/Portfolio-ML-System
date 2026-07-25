"""Training/serving skew: the two paths must produce identical feature frames.

This is the #1 real ML service bug. The structural defence is that
``src/serving/preprocessing.py`` imports the same ``engineer_features`` the trainer
used rather than reimplementing it. This test is what makes that a guarantee
instead of an intention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.split import make_splits
from src.serving.preprocessing import build_features
from src.training.tabular import TabularTrainer


def test_serving_reproduces_the_training_features_row_for_row(registry, trained_churn):
    """Score the same rows through both paths and demand identical matrices."""
    root, problem = trained_churn
    loaded = registry.load(problem)

    trainer = TabularTrainer(problem, sample=True)
    frame = trainer.load_data()
    split = make_splits(frame, {"type": "random", "test_size": 0.25}, "is_attrited", 42)[0]
    training_matrix = trainer.build_matrix(frame, split)["X_test"]
    trainer.finish()

    payloads = frame.iloc[split.test].to_dict(orient="records")[:20]
    serving_rows = [build_features(loaded, payload) for payload in payloads]
    serving_matrix = pd.concat(serving_rows, ignore_index=True)

    assert list(serving_matrix.columns) == list(training_matrix.columns)

    expected = training_matrix.head(20).reset_index(drop=True)
    for column in expected.columns:
        left = pd.to_numeric(serving_matrix[column], errors="coerce")
        right = pd.to_numeric(expected[column], errors="coerce")
        if right.notna().any():
            np.testing.assert_allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=1e-5,
                equal_nan=True,
                err_msg=f"serving/training skew in column {column!r}",
            )


def test_serving_columns_match_the_checkpoint_exactly(registry, trained_churn):
    """Order matters: the model scores positionally once handed a numpy array."""
    _, problem = trained_churn
    loaded = registry.load(problem)
    matrix = build_features(loaded, {"Customer_Age": 45.0, "Credit_Limit": 12000.0})
    assert list(matrix.columns) == loaded.feature_columns


def test_unsupplied_fields_become_nan_not_zero(registry, trained_churn):
    """NaN means 'unknown'; 0 means 'zero'. LightGBM treats them differently."""
    _, problem = trained_churn
    loaded = registry.load(problem)
    matrix = build_features(loaded, {"Customer_Age": 45.0})
    assert matrix["Credit_Limit"].isna().all()


def test_the_target_never_enters_the_serving_frame(registry, trained_churn):
    """Even if a caller sends it. Especially if a caller sends it."""
    _, problem = trained_churn
    loaded = registry.load(problem)
    matrix = build_features(loaded, {"Customer_Age": 45.0, "is_attrited": 1})
    assert "is_attrited" not in matrix.columns
