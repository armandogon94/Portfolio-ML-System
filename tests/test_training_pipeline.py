"""Training pipeline integration tests — end-to-end on tiny data."""

import json
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
