"""The single config-driven trainer, exercised on the CI fixtures."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.config import get_project_root
from src.data.split import make_splits
from src.training.tabular import TabularTrainer, _source_tree_sha256

CONFIGS = ["fraud", "fraud_ulb", "credit_risk", "churn"]


def _fit_once(problem: str):
    trainer = TabularTrainer(problem, sample=True)
    frame = trainer.load_data()
    split = make_splits(
        frame, {"type": "random", "test_size": 0.25}, trainer.config["data"]["target"], 42
    )[0]
    data = trainer.build_matrix(frame, split)
    model = trainer.train(data)
    return trainer, data, model


@pytest.mark.parametrize("problem", CONFIGS)
def test_config_alone_trains_every_problem(problem):
    """One trainer, three problems, zero per-problem branches."""
    trainer, data, model = _fit_once(problem)
    proba = model.predict_proba(data["X_test"])
    assert proba.shape == (len(data["y_test"]), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    trainer.finish()


@pytest.mark.parametrize("problem", CONFIGS)
def test_feature_matrix_excludes_every_denylisted_column(problem):
    trainer, data, _ = _fit_once(problem)
    denylist = set(trainer.config["data"]["denylist"])
    assert not denylist & set(data["X_train"].columns)
    trainer.finish()


@pytest.mark.parametrize("problem", CONFIGS)
def test_configured_split_column_is_never_a_model_feature(problem):
    trainer, data, _ = _fit_once(problem)
    split_column = trainer.config["split"].get("column")
    if split_column is not None:
        assert split_column not in data["X_train"].columns
    trainer.finish()


@pytest.mark.parametrize("problem", CONFIGS)
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

    result = trainer.save_artifacts(object(), {"test_roc_auc": 0.99}, sanity_band_warning=None)

    assert result is None
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "reports").exists()
    trainer.finish()


def test_sample_temporal_run_skips_publication_only_block_spread(monkeypatch):
    trainer = TabularTrainer("fraud", sample=True)
    monkeypatch.setattr(
        trainer,
        "_temporal_spread",
        lambda *args, **kwargs: pytest.fail("sample mode attempted publication uncertainty"),
    )

    metrics = trainer.run()

    assert "test_pr_auc_temporal_block_std" not in metrics


def test_training_source_digest_changes_when_executable_source_changes(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_text("VALUE = 1\n")
    (tmp_path / "configs" / "run.yaml").write_text("seed: 42\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    monkeypatch.setattr("src.training.tabular.get_project_root", lambda: tmp_path)

    before = _source_tree_sha256()
    source.write_text("VALUE = 2\n")

    assert _source_tree_sha256() != before


def test_sample_mode_does_not_write_to_the_production_mlflow_experiment(monkeypatch, tmp_path):
    """A fixture run must not exist in MLflow, even as an untagged partial run."""
    import mlflow

    mlflow.end_run()
    tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    trainer = TabularTrainer("churn", sample=True)
    try:
        assert trainer.mlflow_run_id is None
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        assert client.get_experiment_by_name("fintech-ml-system") is None
    finally:
        trainer.finish()


def test_production_mlflow_run_is_tagged_with_problem_sample_and_config(monkeypatch, tmp_path):
    import mlflow

    mlflow.end_run()
    tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    trainer = TabularTrainer("churn", sample=False)
    try:
        assert trainer.mlflow_run_id is not None
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        tags = client.get_run(trainer.mlflow_run_id).data.tags
        assert tags["problem"] == "churn"
        assert tags["sample"] == "false"
        assert tags["config"] == "churn"
    finally:
        trainer.finish()


def test_a_denylisted_column_reaching_the_matrix_aborts_the_run():
    """The guard is an assertion, not a comment."""
    trainer = TabularTrainer("churn", sample=True)
    trainer.feature_columns = ["Customer_Age", "is_attrited"]
    with pytest.raises(ValueError, match="Denylisted columns reached"):
        trainer._assert_no_denylisted(["is_attrited"])
    trainer.finish()


def test_a_score_above_the_sanity_band_is_flagged_for_investigation():
    trainer = TabularTrainer("credit_risk", sample=True)
    warning = trainer._check_sanity_band({"test_roc_auc": 0.995})
    assert warning is not None
    assert "sanity band" in warning.lower()
    trainer.finish()


def test_a_score_below_the_sanity_band_is_flagged_too():
    trainer = TabularTrainer("credit_risk", sample=True)
    warning = trainer._check_sanity_band({"test_roc_auc": 0.51})
    assert warning is not None
    assert "below" in warning.lower()
    trainer.finish()


def test_a_score_inside_the_band_is_not_flagged():
    trainer = TabularTrainer("credit_risk", sample=True)
    assert trainer._check_sanity_band({"test_roc_auc": 0.71}) is None
    trainer.finish()


def test_ulb_does_not_backfill_a_sanity_range_after_the_temporal_fix():
    trainer = TabularTrainer("fraud_ulb", sample=True)
    try:
        assert "sanity_band" not in trainer.config
        warning = trainer._check_sanity_band({"test_score_std": 0.0})
        assert warning is not None
        assert "DEGENERATE" in warning
    finally:
        trainer.finish()


def test_stratified_kfold_sanity_band_consumes_the_cv_mean_not_a_fold_metric():
    trainer = TabularTrainer("churn", sample=True)
    try:
        assert (
            trainer._check_sanity_band(
                {
                    "fold5_test_roc_auc": 0.40,
                    "cv_roc_auc_mean": 0.90,
                    "cv_roc_auc_std": 0.02,
                }
            )
            is None
        )
    finally:
        trainer.finish()


def test_cv_checkpoint_is_refit_on_all_rows_and_oof_predictions_cover_every_row(
    tmp_path, monkeypatch
):
    trainer = TabularTrainer("churn", sample=True)
    frame = trainer.load_data()
    trainer.sample = False
    trainer.config["model"] = {"type": "prior", "params": {}}
    trainer.config["baselines"] = []
    trainer.config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints" / "churn")
    trainer.config["training"]["reports_dir"] = str(tmp_path / "reports")
    trainer.config["training"]["refit_full_after_cv"] = True
    monkeypatch.setattr(trainer, "load_data", lambda: frame)
    monkeypatch.setattr(
        trainer,
        "train",
        lambda data: DummyClassifier(strategy="prior").fit(data["X_train"], data["y_train"]),
    )

    metrics = trainer.run()

    saved = joblib.load(tmp_path / "checkpoints" / "churn" / "model.joblib")
    metadata = json.loads((tmp_path / "checkpoints" / "churn" / "metadata.json").read_text())
    oof = pd.read_csv(tmp_path / "reports" / "churn_oof_predictions.csv")

    assert saved.training_rows_ == len(frame)
    assert metadata["checkpoint_fit"]["scope"] == "all_rows_refit_after_cross_validation"
    assert metadata["checkpoint_fit"]["n_rows"] == len(frame)
    assert "test_roc_auc" not in metrics
    assert "cv_roc_auc_mean" in metrics
    assert "cv_roc_auc_std" in metrics
    assert sorted(oof["row_index"]) == list(range(len(frame)))
    assert oof["row_index"].is_unique
    assert set(oof["fold"]) == {1, 2, 3, 4, 5}
    assert {"y_true", "score_model"}.issubset(oof.columns)


def test_tracking_failure_cannot_prevent_checkpoint_write(tmp_path, monkeypatch):
    import mlflow

    trainer = TabularTrainer("churn", sample=True)
    frame = trainer.load_data()
    trainer.sample = False
    trainer.use_mlflow = True
    trainer.config["split"] = {"type": "random", "test_size": 0.25}
    trainer.config["model"] = {"type": "prior", "params": {}}
    trainer.config["baselines"] = []
    trainer.config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints" / "churn")
    trainer.config["training"]["reports_dir"] = str(tmp_path / "reports")
    monkeypatch.setattr(trainer, "load_data", lambda: frame)
    monkeypatch.setattr(
        trainer,
        "train",
        lambda data: DummyClassifier(strategy="prior").fit(data["X_train"], data["y_train"]),
    )
    monkeypatch.setattr(
        mlflow,
        "log_metrics",
        lambda metrics: (_ for _ in ()).throw(OSError("blip")),
    )

    trainer.run()

    assert (tmp_path / "checkpoints" / "churn" / "model.joblib").exists()
    assert (tmp_path / "checkpoints" / "churn" / "metadata.json").exists()


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


# ── the tabular path must never import torch ─────────────────────────────────


def test_tabular_training_never_imports_torch():
    """PyTorch and LightGBM cannot share one process on macOS arm64.

    Both ship an OpenMP runtime; loading the two segfaults inside
    ``LGBMClassifier.fit``. Measured on 2026-07-25 against the real 8,101-row
    churn matrix with torch 2.x and LightGBM 4.6.0: without the torch import the
    fit completes, with it the process dies with ``Segmentation fault: 11``.

    This test runs the tabular trainer in a **subprocess** so an earlier test in
    this session cannot have imported torch already, and asserts both that the
    run survives and that torch never entered ``sys.modules``.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        assert "torch" not in sys.modules, "torch was imported before the trainer ran"

        from src.data.split import make_splits
        from src.training.tabular import TabularTrainer

        trainer = TabularTrainer("churn", sample=True)
        frame = trainer.load_data()
        splits = make_splits(
            frame,
            trainer.config["split"],
            trainer.config["data"]["target"],
            trainer.seed,
        )
        data = trainer.build_matrix(frame, splits[0])
        model = trainer.train(data)
        trainer.evaluate(model, data)

        assert "torch" not in sys.modules, (
            "the tabular path imported torch; on macOS arm64 that segfaults "
            "LightGBM. See BaseTrainer.seed_torch."
        )
        print("NO_TORCH_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(get_project_root()),
    )
    assert result.returncode == 0, (
        f"tabular training subprocess failed (returncode {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NO_TORCH_OK" in result.stdout
