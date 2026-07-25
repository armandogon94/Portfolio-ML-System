"""The single config-driven trainer, exercised on the CI fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import make_splits
from src.training.tabular import TabularTrainer

PROBLEMS = ["fraud", "credit_risk", "churn"]


def _fit_once(problem: str):
    trainer = TabularTrainer(problem, sample=True)
    frame = trainer.load_data()
    split = make_splits(
        frame, {"type": "random", "test_size": 0.25}, trainer.config["data"]["target"], 42
    )[0]
    data = trainer.build_matrix(frame, split)
    model = trainer.train(data)
    return trainer, data, model


@pytest.mark.parametrize("problem", PROBLEMS)
def test_config_alone_trains_every_problem(problem):
    """One trainer, three problems, zero per-problem branches."""
    trainer, data, model = _fit_once(problem)
    proba = model.predict_proba(data["X_test"])
    assert proba.shape == (len(data["y_test"]), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    trainer.finish()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_feature_matrix_excludes_every_denylisted_column(problem):
    trainer, data, _ = _fit_once(problem)
    denylist = set(trainer.config["data"]["denylist"])
    assert not denylist & set(data["X_train"].columns)
    trainer.finish()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_train_and_test_matrices_have_identical_columns(problem):
    """Column drift between splits is silent and catastrophic."""
    trainer, data, _ = _fit_once(problem)
    assert list(data["X_train"].columns) == list(data["X_test"].columns)
    trainer.finish()


def test_same_seed_gives_the_same_model():
    first, first_data, first_model = _fit_once("churn")
    second, second_data, second_model = _fit_once("churn")
    np.testing.assert_allclose(
        first_model.predict_proba(first_data["X_test"])[:, 1],
        second_model.predict_proba(second_data["X_test"])[:, 1],
    )
    first.finish()
    second.finish()


def test_sample_mode_refuses_to_write_a_checkpoint(tmp_path):
    """Fixture numbers must not be able to become published numbers by accident."""
    trainer = TabularTrainer("churn", sample=True)
    trainer.config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints" / "churn")
    trainer.config["training"]["reports_dir"] = str(tmp_path / "reports")

    result = trainer.save_artifacts(object(), {"test_roc_auc": 0.99}, leak_warning=None)

    assert result is None
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "reports").exists()
    trainer.finish()


def test_a_denylisted_column_reaching_the_matrix_aborts_the_run():
    """The guard is an assertion, not a comment."""
    trainer = TabularTrainer("churn", sample=True)
    trainer.feature_columns = ["Customer_Age", "is_attrited"]
    with pytest.raises(ValueError, match="Denylisted columns reached"):
        trainer._assert_no_denylisted(["is_attrited"])
    trainer.finish()


def test_a_score_above_the_expected_ceiling_is_flagged_as_leakage():
    """The 0.964 incident in one test: a suspiciously good number must shout."""
    trainer = TabularTrainer("credit_risk", sample=True)
    warning = trainer._check_expected_band({"test_roc_auc": 0.995})
    assert warning is not None
    assert "leakage" in warning.lower()
    trainer.finish()


def test_a_score_below_the_expected_floor_is_flagged_too():
    trainer = TabularTrainer("credit_risk", sample=True)
    warning = trainer._check_expected_band({"test_roc_auc": 0.51})
    assert warning is not None
    assert "below" in warning.lower()
    trainer.finish()


def test_a_score_inside_the_band_is_not_flagged():
    trainer = TabularTrainer("credit_risk", sample=True)
    assert trainer._check_expected_band({"test_roc_auc": 0.71}) is None
    trainer.finish()


def test_baselines_are_evaluated_alongside_the_model():
    """No leaderboard row without a baseline row."""
    trainer, data, model = _fit_once("churn")
    metrics = trainer.evaluate(model, data)
    assert "baseline_prior_pr_auc" in metrics
    assert "baseline_logreg_pr_auc" in metrics
    assert "test_pr_auc_delta" in metrics
    trainer.finish()


def test_feature_alignment_is_order_stable():
    """Serving reindexes to these columns; order matters or the model scores noise."""
    trainer, data, _ = _fit_once("churn")
    reordered = data["X_test"][list(reversed(data["X_test"].columns))]
    aligned = trainer._align(reordered)
    assert list(aligned.columns) == trainer.feature_columns
    pd.testing.assert_frame_equal(aligned, data["X_test"], check_dtype=False)
    trainer.finish()
