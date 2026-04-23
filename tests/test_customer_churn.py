"""Integration tests for the customer-churn slice.

Covers the three layers the agent owns end-to-end:
1. Synthetic generator — shape, columns, churn-rate sanity.
2. Training pipeline — tiny-data run produces checkpoint + metadata + CSV.
3. Predictor — loads checkpoint, returns the documented output contract
   with correct retention-tier thresholds.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import get_project_root
from src.data.generate_customer_churn import generate_customer_churn_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Clone configs into tmp_path so BaseTrainer reads our tuned overrides."""
    real_root = get_project_root()
    configs_dest = tmp_path / "configs"
    shutil.copytree(real_root / "configs", configs_dest)
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestCustomerChurnGenerator:
    def test_has_expected_columns(self):
        df = generate_customer_churn_data(n_samples=500, seed=42)
        expected = {
            "tenure_months", "balance", "num_products", "has_credit_card",
            "is_active_member", "estimated_salary", "age", "geography_tier",
            "churned",
        }
        assert set(df.columns) == expected

    def test_value_ranges(self):
        df = generate_customer_churn_data(n_samples=2000, seed=42)
        assert df["tenure_months"].between(0, 120).all()
        assert df["balance"].between(0, 250_000).all()
        assert df["num_products"].between(1, 6).all()
        assert df["has_credit_card"].isin([0, 1]).all()
        assert df["is_active_member"].isin([0, 1]).all()
        assert df["estimated_salary"].between(10_000, 200_000).all()
        assert df["age"].between(18, 92).all()
        assert df["geography_tier"].isin([1, 2, 3]).all()
        assert df["churned"].isin([0, 1]).all()

    def test_churn_rate_in_realistic_band(self):
        """Synthetic churn should land in a 10–35% band around the 20% target."""
        df = generate_customer_churn_data(n_samples=5000, seed=42)
        assert 0.10 <= df["churned"].mean() <= 0.35


class TestCustomerChurnPipeline:
    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_customer_churn_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "customer_churn.csv")

        checkpoint_dir = project / "checkpoints" / "customer_churn"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_customer_churn import CustomerChurnTrainer

            trainer = CustomerChurnTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        assert (checkpoint_dir / "model.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "customer_churn"
        assert metadata["model_type"] == "xgboost"
        assert "test_accuracy" in metadata["metrics"]
        assert "test_auc_roc" in metadata["metrics"]

        assert (results_dir / "customer_churn_metrics.csv").exists()
        assert "test_accuracy" in metrics


class TestCustomerChurnPredictor:
    """Drive ModelPredictor end-to-end against a tmp checkpoint."""

    def _train_and_patch_root(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_customer_churn_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "customer_churn.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_customer_churn import CustomerChurnTrainer

            trainer = CustomerChurnTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()
        return project

    def test_predict_returns_contract_shape(self, tmp_path):
        project = self._train_and_patch_root(tmp_path)
        with patch("src.config.PROJECT_ROOT", project):
            from src.serving.predictor import ModelPredictor

            predictor = ModelPredictor()
            out = predictor.predict_customer_churn({
                "tenure_months": 48,
                "balance": 50000,
                "num_products": 2,
                "has_credit_card": 1,
                "is_active_member": 1,
                "estimated_salary": 75000,
                "age": 40,
                "geography_tier": 2,
            })

        assert set(out.keys()) == {
            "probability_churn",
            "retention_recommendation",
            "confidence",
        }
        assert 0.0 <= out["probability_churn"] <= 1.0
        assert 0.5 <= out["confidence"] <= 1.0
        assert out["retention_recommendation"] in {
            "URGENT_OUTREACH", "PROACTIVE_CHECKIN", "NO_ACTION",
        }

    def test_recommendation_thresholds(self, tmp_path):
        """Tier boundaries: ≥0.6 URGENT, ≥0.3 PROACTIVE, else NO_ACTION."""
        project = self._train_and_patch_root(tmp_path)
        with patch("src.config.PROJECT_ROOT", project):
            from src.serving.predictor import ModelPredictor

            predictor = ModelPredictor()
            # Force-override the model so we can pin probabilities.

            class StubModel:
                def __init__(self, prob):
                    self._prob = prob

                def predict_proba(self, X):
                    import numpy as np
                    return np.array([[1 - self._prob, self._prob]])

            sample = {
                "tenure_months": 10, "balance": 0, "num_products": 1,
                "has_credit_card": 0, "is_active_member": 0,
                "estimated_salary": 40000, "age": 55, "geography_tier": 3,
            }

            predictor._models["customer_churn"] = StubModel(0.75)
            predictor._artifacts["customer_churn"] = {"metadata": {}}
            assert predictor.predict_customer_churn(sample)[
                "retention_recommendation"
            ] == "URGENT_OUTREACH"

            predictor._models["customer_churn"] = StubModel(0.45)
            assert predictor.predict_customer_churn(sample)[
                "retention_recommendation"
            ] == "PROACTIVE_CHECKIN"

            predictor._models["customer_churn"] = StubModel(0.10)
            assert predictor.predict_customer_churn(sample)[
                "retention_recommendation"
            ] == "NO_ACTION"

    def test_explain_returns_shap_shape(self, tmp_path):
        project = self._train_and_patch_root(tmp_path)
        with patch("src.config.PROJECT_ROOT", project):
            from src.serving.predictor import ModelPredictor

            predictor = ModelPredictor()
            out = predictor.explain_customer_churn({
                "tenure_months": 48,
                "balance": 50000,
                "num_products": 2,
                "has_credit_card": 1,
                "is_active_member": 1,
                "estimated_salary": 75000,
                "age": 40,
                "geography_tier": 2,
            })

        assert out["explanation_type"] == "shap"
        assert set(out["feature_importances"].keys()) == {
            "tenure_months", "balance", "num_products", "has_credit_card",
            "is_active_member", "estimated_salary", "age", "geography_tier",
        }
        assert len(out["top_features"]) >= 1
