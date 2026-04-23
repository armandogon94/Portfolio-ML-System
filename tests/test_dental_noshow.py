"""Phase A.4 — integration test for the dental no-show model pipeline.

Mirrors ``TestCreditRiskPipeline`` — synthesize a small dataset, run the
trainer end-to-end in a tmp project root, and assert checkpoint +
metadata + results CSV are all written.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import get_project_root
from src.data.generate_dental_noshow import generate_dental_noshow_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    real_root = get_project_root()
    shutil.copytree(real_root / "configs", tmp_path / "configs")
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestDentalNoShowGenerator:
    """Sanity checks on the synthetic generator itself."""

    def test_generator_shape_and_target_balance(self):
        df = generate_dental_noshow_data(n_samples=2000, seed=42)

        assert len(df) == 2000
        expected_cols = {
            "age",
            "prior_no_shows",
            "days_until_appointment",
            "appointment_hour",
            "distance_km",
            "insurance_type",
            "procedure_complexity",
            "prior_appointments",
            "is_no_show",
        }
        assert set(df.columns) == expected_cols

        # Feature ranges
        assert df["age"].between(18, 90).all()
        assert df["prior_no_shows"].between(0, 10).all()
        assert df["days_until_appointment"].between(0, 60).all()
        assert df["appointment_hour"].between(8, 18).all()
        assert df["distance_km"].between(0, 50).all()
        assert df["insurance_type"].between(1, 4).all()
        assert df["procedure_complexity"].between(1, 5).all()
        assert df["prior_appointments"].between(0, 20).all()

        # No-show rate should land in a reasonable range (~10-45% with
        # the logistic target + noise).
        rate = df["is_no_show"].mean()
        assert 0.10 < rate < 0.45, f"Unexpected no-show rate: {rate}"


class TestDentalNoShowPipeline:
    """End-to-end trainer run on tiny synthetic data."""

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_dental_noshow_data(n_samples=400, seed=42)
        _write_csv(df, project / "data" / "raw" / "dental_noshow.csv")

        checkpoint_dir = project / "checkpoints" / "dental_noshow"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_dental_noshow import DentalNoShowTrainer
            trainer = DentalNoShowTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        assert (checkpoint_dir / "model.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "dental_noshow"
        assert metadata["model_type"] == "xgboost"
        assert "test_accuracy" in metadata["metrics"]
        assert "test_auc_roc" in metadata["metrics"]

        assert (results_dir / "dental_noshow_metrics.csv").exists()
        assert "test_accuracy" in metrics
        assert "test_auc_roc" in metrics


class TestDentalNoShowPredictor:
    """Predictor-level smoke tests for predict + explain risk banding."""

    def test_predict_and_explain_roundtrip(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_dental_noshow_data(n_samples=600, seed=42)
        _write_csv(df, project / "data" / "raw" / "dental_noshow.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_dental_noshow import DentalNoShowTrainer
            trainer = DentalNoShowTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

            from src.serving.predictor import ModelPredictor
            predictor = ModelPredictor()

            sample = {
                "age": 35,
                "prior_no_shows": 1,
                "days_until_appointment": 14,
                "appointment_hour": 10,
                "distance_km": 8.0,
                "insurance_type": 1,
                "procedure_complexity": 2,
                "prior_appointments": 5,
            }
            result = predictor.predict_dental_noshow(sample)

        assert 0.0 <= result["probability_no_show"] <= 1.0
        assert result["risk_band"] in {"HIGH_RISK", "MODERATE", "LIKELY_TO_SHOW"}
        assert 0.5 <= result["confidence"] <= 1.0

        # Risk-band threshold logic
        p = result["probability_no_show"]
        if p >= 0.40:
            assert result["risk_band"] == "HIGH_RISK"
        elif p <= 0.15:
            assert result["risk_band"] == "LIKELY_TO_SHOW"
        else:
            assert result["risk_band"] == "MODERATE"

        # Explanation contract
        with patch("src.config.PROJECT_ROOT", project):
            explanation = predictor.explain_dental_noshow(sample)
        assert explanation["explanation_type"] == "shap"
        assert "feature_importances" in explanation
        assert set(explanation["feature_importances"].keys()) == {
            "age",
            "prior_no_shows",
            "days_until_appointment",
            "appointment_hour",
            "distance_km",
            "insurance_type",
            "procedure_complexity",
            "prior_appointments",
        }
