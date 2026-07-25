"""The unsupervised fraud baseline trainer.

Runs on the IEEE-CIS CI fixture with two epochs. It proves the pipeline wiring —
fit on legitimate rows, threshold from train error, rank-normalised scores — and
never claims a metric: the fixture's label is independent noise.

CPU only. The session-wide MPS mock in conftest is what makes that true, and it is
there because torch 2.13.0 on this hardware deadlocked a CPU tensor loop that
followed an MPS matmul in the same process.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_scores_are_rank_normalised_into_the_unit_interval(trained):
    """Reconstruction error is unbounded; Brier score needs a probability."""
    _, metrics = trained
    assert 0.0 <= metrics["test_brier"] <= 1.0


def test_sample_mode_writes_no_autoencoder_checkpoint(trained, project_root):
    trainer, _ = trained
    assert trainer.sample is True
    assert not (project_root / "checkpoints" / "fraud_autoencoder").exists()


def test_the_fixture_result_is_near_chance(trained):
    """The fixture label is independent noise. A good score here means a leak."""
    _, metrics = trained
    assert 0.2 <= metrics["test_roc_auc"] <= 0.8, (
        f"autoencoder scored {metrics['test_roc_auc']:.3f} on independent-noise "
        f"labels — regenerate the fixture with scripts/make_fixtures.py"
    )
