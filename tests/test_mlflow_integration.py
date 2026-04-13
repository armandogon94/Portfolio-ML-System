"""MLflow integration tests — verify BaseTrainer logs to MLflow."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest

from src.config import get_project_root
from src.data.generate_credit_risk import generate_credit_risk_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    shutil.copytree(real_root / "configs", tmp_path / "configs")
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_mlflow_state():
    """Ensure no active MLflow run leaks between tests."""
    yield
    if mlflow.active_run():
        mlflow.end_run()


class TestMLflowInit:

    def test_trainer_has_mlflow_flag(self, tmp_path):
        """BaseTrainer should expose use_mlflow attribute."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = str(tmp_path / "mlruns")
        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            assert hasattr(trainer, "use_mlflow")
            assert isinstance(trainer.use_mlflow, bool)
            trainer.finish()

    def test_mlflow_run_started(self, tmp_path):
        """An MLflow run should be active after trainer init."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = str(tmp_path / "mlruns")
        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            assert trainer.use_mlflow is True
            assert mlflow.active_run() is not None
            trainer.finish()

    def test_mlflow_fallback_on_error(self, tmp_path):
        """Trainer should gracefully fall back when MLflow init raises."""
        project = _setup_tmp_project(tmp_path)
        with patch("src.config.PROJECT_ROOT", project), \
             patch("src.training.trainer.mlflow.start_run",
                   side_effect=Exception("Connection refused")):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            assert trainer.use_mlflow is False
            trainer.finish()


class TestMLflowMetricLogging:

    def test_metrics_logged_to_mlflow(self, tmp_path):
        """Metrics logged via log_metric should appear in MLflow."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = str(tmp_path / "mlruns")
        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.log_metric("test_accuracy", 0.95)
            trainer.log_metrics({"test_f1": 0.88, "test_auc": 0.92})

            run_id = mlflow.active_run().info.run_id
            trainer.finish()

        client = mlflow.tracking.MlflowClient(tracking_uri)
        run_data = client.get_run(run_id)
        logged = run_data.data.metrics
        assert logged["test_accuracy"] == pytest.approx(0.95)
        assert logged["test_f1"] == pytest.approx(0.88)
        assert logged["test_auc"] == pytest.approx(0.92)

    def test_config_logged_as_params(self, tmp_path):
        """Trainer config should be logged as MLflow params."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = str(tmp_path / "mlruns")
        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            run_id = mlflow.active_run().info.run_id
            trainer.finish()

        client = mlflow.tracking.MlflowClient(tracking_uri)
        run_data = client.get_run(run_id)
        params = run_data.data.params
        assert params["problem"] == "credit_risk"
        assert "model.type" in params


class TestMLflowEndToEnd:

    def test_full_pipeline_logs_to_mlflow(self, tmp_path):
        """A full trainer.run() should create an MLflow run with metrics."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = str(tmp_path / "mlruns")

        df = generate_credit_risk_data(n_samples=200, seed=42)
        raw_dir = project / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "credit_risk.csv", index=False)

        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            run_id = mlflow.active_run().info.run_id
            trainer.run()

        client = mlflow.tracking.MlflowClient(tracking_uri)
        run_data = client.get_run(run_id)
        assert run_data.data.metrics.get("test_accuracy") is not None
        assert run_data.info.status == "FINISHED"


class TestMLflowModelRegistry:

    def test_model_registered_after_run(self, tmp_path):
        """After trainer.run(), model should be in the MLflow registry."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

        df = generate_credit_risk_data(n_samples=200, seed=42)
        raw_dir = project / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "credit_risk.csv", index=False)

        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        client = mlflow.tracking.MlflowClient(tracking_uri)
        models = client.search_registered_models()
        model_names = [m.name for m in models]
        assert "credit_risk" in model_names

        versions = client.search_model_versions("name='credit_risk'")
        assert len(versions) >= 1

    def test_metadata_contains_mlflow_fields(self, tmp_path):
        """metadata.json should include mlflow_run_id after training."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

        df = generate_credit_risk_data(n_samples=200, seed=42)
        raw_dir = project / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "credit_risk.csv", index=False)

        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer
            trainer = CreditRiskTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

        metadata_path = project / "checkpoints" / "credit_risk" / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert "mlflow_run_id" in metadata
        assert "mlflow_model_version" in metadata
        assert metadata["mlflow_run_id"] is not None

    def test_retraining_increments_version(self, tmp_path):
        """Re-training should create a new version in the registry."""
        project = _setup_tmp_project(tmp_path)
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

        df = generate_credit_risk_data(n_samples=200, seed=42)
        raw_dir = project / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / "credit_risk.csv", index=False)

        with patch("src.config.PROJECT_ROOT", project), \
             patch.dict("os.environ", {"MLFLOW_TRACKING_URI": tracking_uri}):
            from src.training.train_credit_risk import CreditRiskTrainer

            # First training
            t1 = CreditRiskTrainer(use_wandb=False)
            t1.config["model"]["params"]["n_estimators"] = 5
            t1.run()

            # Second training
            t2 = CreditRiskTrainer(use_wandb=False)
            t2.config["model"]["params"]["n_estimators"] = 5
            t2.run()

        client = mlflow.tracking.MlflowClient(tracking_uri)
        versions = client.search_model_versions("name='credit_risk'")
        assert len(versions) >= 2
