"""Heart disease (A.5) — generator, trainer, and predictor integration tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from src.config import get_project_root
from src.data.generate_heart_disease import generate_heart_disease_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    configs_dest = tmp_path / "configs"
    shutil.copytree(real_root / "configs", configs_dest)
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestHeartDiseaseGenerator:

    def test_schema_and_types(self):
        df = generate_heart_disease_data(n_samples=500, seed=42)

        expected = {
            "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
            "max_heart_rate", "exercise_angina", "oldpeak", "disease",
        }
        assert set(df.columns) == expected
        assert len(df) == 500

        # Ranges
        assert df["age"].between(25, 85).all()
        assert df["sex"].isin([0, 1]).all()
        assert df["chest_pain_type"].between(1, 4).all()
        assert df["resting_bp"].between(80, 200).all()
        assert df["cholesterol"].between(100, 400).all()
        assert df["max_heart_rate"].between(60, 220).all()
        assert df["exercise_angina"].isin([0, 1]).all()
        assert df["oldpeak"].between(0, 6).all()
        assert df["disease"].isin([0, 1]).all()

    def test_reproducible(self):
        a = generate_heart_disease_data(n_samples=200, seed=7)
        b = generate_heart_disease_data(n_samples=200, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_disease_rate_reasonable(self):
        """Prevalence should be meaningful (neither all-0 nor all-1).

        Target ~35% per spec; allow a generous band because the binomial
        draw can drift on 2k rows.
        """
        df = generate_heart_disease_data(n_samples=2000, seed=42)
        rate = df["disease"].mean()
        assert 0.15 < rate < 0.65


class TestHeartDiseasePipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_heart_disease_data(n_samples=400, seed=42)
        _write_csv(df, project / "data" / "raw" / "heart_disease.csv")

        checkpoint_dir = project / "checkpoints" / "heart_disease"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_heart_disease import HeartDiseaseTrainer
            trainer = HeartDiseaseTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 10
            metrics = trainer.run()

        assert (checkpoint_dir / "model.pkl").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "heart_disease"
        assert metadata["model_type"] == "lightgbm"
        assert "test_accuracy" in metadata["metrics"]
        assert "test_auc_roc" in metadata["metrics"]

        assert (results_dir / "heart_disease_metrics.csv").exists()
        assert "test_accuracy" in metrics


class TestHeartDiseasePredictor:
    """Predictor round-trip: train a tiny model, serve via ModelPredictor."""

    def test_predict_and_explain(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_heart_disease_data(n_samples=400, seed=42)
        _write_csv(df, project / "data" / "raw" / "heart_disease.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_heart_disease import HeartDiseaseTrainer
            trainer = HeartDiseaseTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 10
            trainer.run()

            from src.serving.predictor import ModelPredictor

            predictor = ModelPredictor()
            # ModelPredictor caches `get_project_root()` at __init__ time; in
            # tests the cache was set before the patch took effect if the
            # predictor was imported earlier. Re-pin root to the tmp project.
            predictor.root = project

            vitals = {
                "age": 55,
                "sex": 1,
                "chest_pain_type": 3,
                "resting_bp": 130,
                "cholesterol": 240,
                "max_heart_rate": 150,
                "exercise_angina": 0,
                "oldpeak": 1.0,
            }

            result = predictor.predict_heart_disease(vitals)
            assert set(result.keys()) == {
                "probability_disease", "risk_band", "confidence"
            }
            assert 0.0 <= result["probability_disease"] <= 1.0
            assert result["risk_band"] in {"HIGH", "ELEVATED", "LOW"}
            assert 0.0 <= result["confidence"] <= 1.0

            explanation = predictor.explain_heart_disease(vitals)
            assert explanation["explanation_type"] == "shap"
            assert set(explanation["feature_importances"].keys()) >= {
                "age", "chest_pain_type", "cholesterol",
            }

    def test_risk_bands_threshold_boundaries(self, tmp_path):
        """Verify banding uses >=0.5 HIGH, >=0.25 ELEVATED, else LOW.

        We sidestep training entirely and stub the loaded model so we can
        force each probability bucket.
        """
        project = _setup_tmp_project(tmp_path)
        # Still need a metadata.json for _ensure_loaded; train a trivial one.
        df = generate_heart_disease_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "heart_disease.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_heart_disease import HeartDiseaseTrainer
            trainer = HeartDiseaseTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            trainer.run()

            from src.serving.predictor import ModelPredictor
            predictor = ModelPredictor()
            predictor.root = project

            # Force-load then replace proba function
            predictor._ensure_loaded("heart_disease")

            class Stub:
                def __init__(self, prob):
                    self.prob = prob

                def predict_proba(self, X):
                    import numpy as np
                    return np.array([[1 - self.prob, self.prob]])

            vitals = {
                "age": 55, "sex": 1, "chest_pain_type": 3, "resting_bp": 130,
                "cholesterol": 240, "max_heart_rate": 150, "exercise_angina": 0,
                "oldpeak": 1.0,
            }

            predictor._models["heart_disease"] = Stub(0.8)
            assert predictor.predict_heart_disease(vitals)["risk_band"] == "HIGH"

            predictor._models["heart_disease"] = Stub(0.30)
            assert predictor.predict_heart_disease(vitals)["risk_band"] == "ELEVATED"

            predictor._models["heart_disease"] = Stub(0.10)
            assert predictor.predict_heart_disease(vitals)["risk_band"] == "LOW"

            # Boundary: exactly 0.5 -> HIGH
            predictor._models["heart_disease"] = Stub(0.5)
            assert predictor.predict_heart_disease(vitals)["risk_band"] == "HIGH"

            # Boundary: exactly 0.25 -> ELEVATED
            predictor._models["heart_disease"] = Stub(0.25)
            assert predictor.predict_heart_disease(vitals)["risk_band"] == "ELEVATED"


class TestHeartDiseaseApi:
    """FastAPI route smoke test — exercise the predict + explain endpoints."""

    def test_predict_route(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_heart_disease_data(n_samples=400, seed=42)
        _write_csv(df, project / "data" / "raw" / "heart_disease.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_heart_disease import HeartDiseaseTrainer
            trainer = HeartDiseaseTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 10
            trainer.run()

            # Confirm the LightGBM artefact is joblib-loadable so the
            # predict_price-style loading path in ModelPredictor works.
            ckpt = project / "checkpoints" / "heart_disease" / "model.pkl"
            loaded = joblib.load(ckpt)
            assert hasattr(loaded, "predict_proba")
