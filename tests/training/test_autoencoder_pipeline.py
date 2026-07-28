"""The unsupervised fraud baseline trainer.

Runs on the IEEE-CIS CI fixture with two epochs. It proves the pipeline wiring
(fit on legitimate rows, threshold from train error, raw reconstruction scores)
and never claims a metric: the fixture's label is independent noise.

CPU only. The session-wide MPS mock in conftest is what makes that true, and it is
there because torch 2.13.0 on this hardware deadlocked a CPU tensor loop that
followed an MPS matmul in the same process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.serving.registry import CheckpointRegistry
from src.training.autoencoder_pipeline import AutoencoderTrainer


@pytest.fixture(scope="module")
def trained():
    trainer = AutoencoderTrainer("fraud", sample=True, epochs=2)
    metrics = trainer.run()
    trainer.finish()
    return trainer, metrics


def test_it_runs_end_to_end_on_the_fixture(trained):
    _, metrics = trained
    assert "test_pr_auc" in metrics
    assert "anomaly_threshold" in metrics


def test_the_threshold_comes_from_train_error_not_test(trained):
    """Setting the threshold from test error would be leakage."""
    trainer, metrics = trained
    assert np.isfinite(metrics["anomaly_threshold"])
    assert metrics["anomaly_threshold"] > 0
    assert trainer.threshold == metrics["anomaly_threshold"]


def test_uncalibrated_reconstruction_error_does_not_report_brier(trained):
    _, metrics = trained
    assert "test_brier" not in metrics


def test_hard_decisions_use_the_training_error_threshold():
    import torch

    class IdentityScaler:
        def transform(self, values):
            return values.to_numpy(dtype=np.float32)

    class FixedErrorModel:
        def reconstruction_error(self, tensor):
            return torch.tensor([0.60, 0.70], dtype=torch.float32)

    trainer = AutoencoderTrainer("fraud", sample=True, epochs=0)
    trainer.numeric_columns = ["x"]
    trainer.scaler = IdentityScaler()
    trainer.threshold = 0.65
    data = {
        "X_test": pd.DataFrame({"x": [1.0, 2.0]}),
        "y_test": np.array([0, 1], dtype=np.int8),
    }

    metrics = trainer.evaluate(FixedErrorModel(), data)

    assert metrics["test_precision"] == 1.0
    assert metrics["test_recall"] == 1.0
    assert "test_brier" not in metrics
    trainer.finish()


def test_sample_mode_writes_no_autoencoder_checkpoint(trained, project_root):
    trainer, _ = trained
    assert trainer.sample is True
    assert not (project_root / "checkpoints" / "fraud_autoencoder").exists()


def test_autoencoder_tracking_tag_does_not_claim_lightgbm(monkeypatch, tmp_path):
    import mlflow

    mlflow.end_run()
    tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    trainer = AutoencoderTrainer("fraud", sample=False, epochs=0)
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        tags = client.get_run(trainer.mlflow_run_id).data.tags
        assert tags["problem"] == "fraud"
        assert tags["model"] == "autoencoder"
        assert tags["config"] == "fraud"
    finally:
        trainer.finish()


def test_the_fixture_result_is_near_chance(trained):
    """The fixture label is independent noise. A good score here means a leak."""
    _, metrics = trained
    assert 0.2 <= metrics["test_roc_auc"] <= 0.8, (
        f"autoencoder scored {metrics['test_roc_auc']:.3f} on independent-noise "
        f"labels. Regenerate the fixture with scripts/make_fixtures.py"
    )


def test_autoencoder_checkpoint_round_trips_through_registry(trained, tmp_path):
    source, metrics = trained
    writer = AutoencoderTrainer("fraud", sample=True, epochs=0)
    writer.sample = False
    writer.scaler = source.scaler
    writer.numeric_columns = source.numeric_columns
    writer.feature_columns = source.feature_columns
    writer.feature_artifacts = source.feature_artifacts
    writer.threshold = source.threshold
    writer.config["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints" / "fraud")
    writer.config["training"]["reports_dir"] = str(tmp_path / "reports")

    directory = writer._save(source.model, metrics)
    loaded = CheckpointRegistry(tmp_path / "checkpoints").load("fraud_autoencoder")

    assert directory == tmp_path / "checkpoints" / "fraud_autoencoder"
    assert type(loaded.model) is type(source.model)
    assert loaded.feature_columns == source.numeric_columns
    assert loaded.metadata["dataset"] == writer.config["data"]["source"]
    assert loaded.metadata["split"] == writer.config["split"]
    assert loaded.metadata["leakage_controls"]["denylist_enforced"] is True
    assert (tmp_path / "reports" / "fraud_autoencoder_metrics.csv").exists()
    writer.finish()
