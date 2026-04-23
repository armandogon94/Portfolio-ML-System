"""Integration tests for the delivery ETA pipeline (Phase A.7 — Logistics).

Covers: synthetic generator shape + target bounds, feature engineering
pass-through, full trainer end-to-end on a tiny dataset, and predictor
lazy-load + confidence-interval heuristic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import get_project_root
from src.data.generate_delivery_eta import generate_delivery_eta_data
from src.features.delivery_eta_features import engineer_features, get_feature_columns


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    shutil.copytree(real_root / "configs", tmp_path / "configs")
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestGenerator:
    def test_generator_emits_all_columns(self):
        df = generate_delivery_eta_data(n_samples=500, seed=0)
        expected = set(get_feature_columns()) | {"eta_hours"}
        assert set(df.columns) == expected

    def test_eta_within_expected_bounds(self):
        df = generate_delivery_eta_data(n_samples=2000, seed=1)
        assert df["eta_hours"].between(2.0, 120.0).all()

    def test_feature_ranges_match_spec(self):
        df = generate_delivery_eta_data(n_samples=2000, seed=1)
        assert df["distance_km"].between(1, 2000).all()
        assert df["package_weight_kg"].between(0.1, 50).all()
        assert df["traffic_congestion"].between(1, 5).all()
        assert df["weather_severity"].between(0, 4).all()
        assert df["time_of_day"].between(0, 23).all()
        assert df["day_of_week"].between(0, 6).all()
        assert df["carrier_priority"].between(1, 3).all()
        assert df["origin_destination_tier"].between(1, 4).all()


class TestFeatures:
    def test_feature_columns_stable(self):
        cols = get_feature_columns()
        assert len(cols) == 8
        assert "distance_km" in cols
        assert "origin_destination_tier" in cols

    def test_engineer_features_is_passthrough(self):
        df = generate_delivery_eta_data(n_samples=100, seed=2)
        out = engineer_features(df)
        # Pass-through: columns preserved, values identical, but new object
        pd.testing.assert_frame_equal(out, df)
        assert out is not df


class TestTrainerPipeline:
    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_delivery_eta_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "delivery_eta.csv")

        checkpoint_dir = project / "checkpoints" / "delivery_eta"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_delivery_eta import DeliveryEtaTrainer
            trainer = DeliveryEtaTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 10
            metrics = trainer.run()

        assert (checkpoint_dir / "model.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()
        assert (results_dir / "delivery_eta_metrics.csv").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "delivery_eta"
        assert metadata["model_type"] == "xgboost"
        assert "test_rmse" in metadata["metrics"]
        assert "test_r2" in metadata["metrics"]
        assert "test_rmse" in metrics


class TestPredictor:
    def test_predict_delivery_eta_returns_confidence_interval(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_delivery_eta_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "delivery_eta.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_delivery_eta import DeliveryEtaTrainer
            trainer = DeliveryEtaTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 10
            trainer.run()

            from src.serving.predictor import ModelPredictor
            predictor = ModelPredictor()
            result = predictor.predict_delivery_eta({
                "distance_km": 50.0,
                "package_weight_kg": 2.0,
                "traffic_congestion": 3,
                "weather_severity": 1,
                "time_of_day": 10,
                "day_of_week": 2,
                "carrier_priority": 2,
                "origin_destination_tier": 2,
            })

        assert "eta_hours" in result
        assert "confidence_interval" in result
        eta = result["eta_hours"]
        lo, hi = result["confidence_interval"]
        # ±20% band heuristic defined in the Phase A.7 spec
        assert lo == eta * 0.8
        assert hi == eta * 1.2
        assert 2.0 <= eta <= 120.0
