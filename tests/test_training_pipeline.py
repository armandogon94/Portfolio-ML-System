"""Training pipeline integration tests — end-to-end on tiny data."""

import json
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from src.config import get_project_root
from src.data.generate_credit_risk import generate_credit_risk_data
from src.data.generate_fraud import generate_fraud_data
from src.data.generate_housing import generate_housing_data
from src.data.generate_timeseries import generate_timeseries_data

# Force CPU for all training pipeline tests to avoid MPS hangs in CI/test
_CPU = torch.device("cpu")


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    configs_dest = tmp_path / "configs"
    shutil.copytree(real_root / "configs", configs_dest)
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestTrainerLogging:
    """Tests that BaseTrainer emits structured log messages during training."""

    def test_trainer_logs_pipeline_steps(self, tmp_path, caplog):
        """BaseTrainer.run() logs loading, preprocessing, training, evaluating, saving."""
        project = _setup_tmp_project(tmp_path)
        df = generate_credit_risk_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "credit_risk.csv")

        with caplog.at_level(logging.INFO, logger="src.training.trainer"), \
             patch("src.config.PROJECT_ROOT", project):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 3
            trainer.run()

        messages = " ".join(r.message.lower() for r in caplog.records)
        assert "loading" in messages or "load" in messages
        assert "train" in messages
        assert "evaluat" in messages
        assert "saving" in messages or "checkpoint" in messages or "saved" in messages


class TestCreditRiskPipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_credit_risk_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "credit_risk.csv")

        checkpoint_dir = project / "checkpoints" / "credit_risk"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        assert (checkpoint_dir / "model.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "credit_risk"
        assert metadata["model_type"] == "xgboost"
        assert "test_accuracy" in metadata["metrics"]
        assert "test_auc_roc" in metadata["metrics"]

        assert (results_dir / "credit_risk_metrics.csv").exists()
        assert "test_accuracy" in metrics

    def test_no_wandb_calls(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_credit_risk_data(n_samples=100, seed=42)
        _write_csv(df, project / "data" / "raw" / "credit_risk.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 3
            assert trainer.use_wandb is False
            trainer.run()


class TestFraudDetectionPipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_fraud_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "fraud_transactions.csv")

        checkpoint_dir = project / "checkpoints" / "fraud_detection"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project), \
             patch("src.device.get_device", return_value=_CPU):
            from src.training.train_fraud import FraudDetectionTrainer
            trainer = FraudDetectionTrainer(use_wandb=False)
            trainer.device = _CPU
            trainer.config["model"]["params"]["epochs"] = 2
            trainer.config["model"]["params"]["batch_size"] = 64
            metrics = trainer.run()

        assert (checkpoint_dir / "autoencoder.pt").exists()
        assert (checkpoint_dir / "scaler.pkl").exists()
        assert (checkpoint_dir / "isolation_forest.pkl").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "fraud_detection"
        assert "anomaly_threshold" in metadata["metrics"]

        assert (results_dir / "fraud_detection_metrics.csv").exists()
        assert "anomaly_threshold" in metrics


class TestPricePredictionPipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        checkpoint_dir = project / "checkpoints" / "price_prediction"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        assert (checkpoint_dir / "model.pkl").exists()
        assert (checkpoint_dir / "metadata.json").exists()
        assert "test_rmse" in metrics
        assert "test_r2" in metrics
        assert (results_dir / "price_prediction_metrics.csv").exists()


class TestPriceModalities:
    """Phase A.1.7 — price_prediction must train in synthetic / stream / mixed modalities."""

    def _tiny_zillow_csv(self, path: Path) -> None:
        """Write a tiny Zillow-shaped CSV the housing_adapter can map."""
        import numpy as np

        rng = np.random.default_rng(7)
        n = 150
        df = pd.DataFrame({
            "SquareFootage": rng.integers(800, 4000, n),
            "Bedrooms": rng.integers(1, 6, n),
            "Bathrooms": rng.integers(1, 4, n),
            "YearBuilt": rng.integers(1960, 2024, n),
            "LotSize": rng.integers(2000, 20000, n),
            "GarageSpaces": rng.integers(0, 3, n),
            "HasPool": rng.integers(0, 2, n),
            "NeighborhoodTier": rng.integers(1, 6, n),
            "DistanceToCBD": rng.uniform(1, 30, n).round(1),
            "SalePrice": rng.integers(150_000, 900_000, n),
            "IrrelevantFluff": ["x"] * n,  # extra column the adapter should drop
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def test_synthetic_modality_writes_modality_suffixed_checkpoint(self, tmp_path):
        """modality='synthetic' writes _synthetic checkpoint + metadata.modality."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="synthetic")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        ckpt = project / "checkpoints" / "price_prediction_synthetic"
        assert (ckpt / "model.pkl").exists()
        assert (ckpt / "metadata.json").exists()

        with open(ckpt / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["modality"] == "synthetic"

    def test_stream_modality_uses_kaggle_and_adapter(self, tmp_path, monkeypatch):
        """modality='stream' → fetches Kaggle (mocked), adapts, writes _stream checkpoint."""
        project = _setup_tmp_project(tmp_path)
        zillow_dir = tmp_path / "zillow_cache"
        zillow_dir.mkdir()
        self._tiny_zillow_csv(zillow_dir / "zillow_sales.csv")

        import sys
        from types import SimpleNamespace
        fake_kagglehub = SimpleNamespace(dataset_download=lambda slug: str(zillow_dir))
        monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="stream")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        ckpt = project / "checkpoints" / "price_prediction_stream"
        assert (ckpt / "model.pkl").exists()
        assert (ckpt / "metadata.json").exists()

        with open(ckpt / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["modality"] == "stream"

    def test_mixed_modality_concatenates_both_sources(self, tmp_path, monkeypatch):
        """modality='mixed' → synthetic + adapted stream, modality-suffixed checkpoint."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        zillow_dir = tmp_path / "zillow_cache"
        zillow_dir.mkdir()
        self._tiny_zillow_csv(zillow_dir / "zillow_sales.csv")

        import sys
        from types import SimpleNamespace
        monkeypatch.setitem(
            sys.modules,
            "kagglehub",
            SimpleNamespace(dataset_download=lambda slug: str(zillow_dir)),
        )
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="mixed")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        ckpt = project / "checkpoints" / "price_prediction_mixed"
        assert (ckpt / "model.pkl").exists()

        with open(ckpt / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["modality"] == "mixed"

    def test_synthetic_modality_mirrors_to_legacy_path(self, tmp_path):
        """A.1.8 — modality='synthetic' dual-writes to legacy checkpoints/<problem>/.

        The ModelPredictor in main still reads from the legacy path; mirroring
        the synthetic checkpoint there is how we keep serving backward-compatible
        during the migration.
        """
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="synthetic")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        modality_ckpt = project / "checkpoints" / "price_prediction_synthetic"
        legacy_ckpt = project / "checkpoints" / "price_prediction"

        # Both paths must contain identical artifacts
        assert (modality_ckpt / "model.pkl").exists()
        assert (legacy_ckpt / "model.pkl").exists()
        assert (legacy_ckpt / "metadata.json").exists()

        # Bytes-identical mirror — the predictor must get the exact same model
        assert (modality_ckpt / "model.pkl").read_bytes() == (
            legacy_ckpt / "model.pkl"
        ).read_bytes()

    def test_stream_modality_does_not_mirror_legacy_path(
        self, tmp_path, monkeypatch
    ):
        """A.1.8 — modality='stream' MUST NOT overwrite the legacy checkpoints dir.

        The legacy path belongs to 'synthetic' demos; stream/mixed variants
        live only under their _<modality>/ dirs so they can't clobber the
        default demo predictor.
        """
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        # Seed the legacy path with a "previous synthetic run" sentinel so we
        # can detect whether stream modality overwrites it.
        legacy_ckpt = project / "checkpoints" / "price_prediction"
        legacy_ckpt.mkdir(parents=True, exist_ok=True)
        sentinel = legacy_ckpt / "model.pkl"
        sentinel.write_bytes(b"sentinel-legacy-bytes")

        zillow_dir = tmp_path / "zillow_cache"
        zillow_dir.mkdir()
        self._tiny_zillow_csv(zillow_dir / "zillow_sales.csv")

        import sys
        from types import SimpleNamespace
        monkeypatch.setitem(
            sys.modules,
            "kagglehub",
            SimpleNamespace(dataset_download=lambda slug: str(zillow_dir)),
        )
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="stream")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        stream_ckpt = project / "checkpoints" / "price_prediction_stream"
        assert (stream_ckpt / "model.pkl").exists()
        # Legacy path untouched — sentinel preserved
        assert sentinel.read_bytes() == b"sentinel-legacy-bytes"

    def test_synthetic_mirror_emits_deprecation_log(self, tmp_path, caplog):
        """A.1.8 — mirroring should log a deprecation note pointing to Phase A.9."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        with caplog.at_level(logging.INFO, logger="src.training.trainer"), \
             patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False, modality="synthetic")
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        messages = " ".join(r.message.lower() for r in caplog.records)
        assert "legacy" in messages or "mirror" in messages
        assert "a.9" in messages or "deprec" in messages or "phase" in messages

    def test_no_modality_uses_legacy_checkpoint_path(self, tmp_path):
        """modality=None (default) preserves legacy checkpoints/price_prediction/ path."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_price import PricePredictionTrainer
            trainer = PricePredictionTrainer(use_wandb=False)  # no modality arg
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        # Legacy path — no _synthetic suffix
        ckpt = project / "checkpoints" / "price_prediction"
        assert (ckpt / "model.pkl").exists()
        assert not (project / "checkpoints" / "price_prediction_synthetic").exists()

    def test_mlflow_nested_run_used_when_parent_active(self, tmp_path, monkeypatch):
        """When a parent MLflow run is active, child trainer uses nested=True + modality tag."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "housing.csv")

        start_run_calls = []
        set_tag_calls = []

        class FakeRun:
            def __init__(self, run_id="fake-run"):
                self.info = SimpleNamespace_local(run_id=run_id)

        from types import SimpleNamespace as SimpleNamespace_local

        import mlflow
        # Force MLflow path to activate
        monkeypatch.setenv("WANDB_API_KEY", "")  # ensure wandb disabled

        original_start_run = mlflow.start_run
        original_set_tag = mlflow.set_tag
        original_active_run = mlflow.active_run

        # Pretend a parent run is already active when _init_mlflow fires.
        # _init_mlflow calls mlflow.active_run() BEFORE mlflow.start_run(),
        # so the parent must be present from the first call.
        fake_parent = FakeRun("parent-123")

        def fake_active_run():
            return fake_parent

        def fake_start_run(*args, **kwargs):
            start_run_calls.append(kwargs)
            return FakeRun(f"child-{len(start_run_calls)}")

        def fake_set_tag(key, value):
            set_tag_calls.append((key, value))

        monkeypatch.setattr(mlflow, "active_run", fake_active_run)
        monkeypatch.setattr(mlflow, "start_run", fake_start_run)
        monkeypatch.setattr(mlflow, "set_tag", fake_set_tag)
        monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
        monkeypatch.setattr(mlflow, "set_experiment", lambda _name: None)
        monkeypatch.setattr(mlflow, "log_params", lambda _params: None)

        try:
            with patch("src.config.PROJECT_ROOT", project):
                from src.training.train_price import PricePredictionTrainer
                trainer = PricePredictionTrainer(use_wandb=False, modality="synthetic")
                # We only care about MLflow init behavior; don't run full pipeline
                # (avoid log_artifacts etc. which would need more mocking)
                _ = trainer
        finally:
            # Restore
            monkeypatch.setattr(mlflow, "active_run", original_active_run)
            monkeypatch.setattr(mlflow, "start_run", original_start_run)
            monkeypatch.setattr(mlflow, "set_tag", original_set_tag)

        # First start_run call was for the child; nested should be True because
        # fake_active_run returned a parent BEFORE any start_run happened.
        assert len(start_run_calls) >= 1
        assert start_run_calls[0].get("nested") is True
        # modality tag applied
        assert ("modality", "synthetic") in set_tag_calls


class TestDemandForecastPipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_timeseries_data(n_years=1, seed=42)
        _write_csv(df, project / "data" / "raw" / "daily_demand.csv")

        checkpoint_dir = project / "checkpoints" / "demand_forecasting"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project), \
             patch("src.device.get_device", return_value=_CPU):
            from src.training.train_forecaster import DemandForecastTrainer
            trainer = DemandForecastTrainer(use_wandb=False)
            trainer.device = _CPU
            trainer.config["model"]["params"]["epochs"] = 2
            trainer.config["model"]["params"]["hidden_size"] = 16
            trainer.config["model"]["params"]["num_layers"] = 1
            metrics = trainer.run()

        assert (checkpoint_dir / "lstm.pt").exists()
        assert (checkpoint_dir / "scalers.pkl").exists()
        assert (checkpoint_dir / "metadata.json").exists()
        assert "test_avg_mae" in metrics
        assert (results_dir / "demand_forecasting_metrics.csv").exists()
