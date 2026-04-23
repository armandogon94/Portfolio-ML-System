"""Rental-price training pipeline integration test (Phase A.3 — Real Estate).

Mirrors ``test_training_pipeline.py::TestPricePredictionPipeline`` — end-to-end
on tiny data (200 samples, 5 boosting rounds) to keep the suite fast while
still exercising the full trainer / checkpoint / results-CSV path.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import get_project_root
from src.data.generate_rental_price import generate_rental_price_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    configs_dest = tmp_path / "configs"
    shutil.copytree(real_root / "configs", configs_dest)
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestRentalPricePipeline:

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_rental_price_data(n_samples=200, seed=42)
        _write_csv(df, project / "data" / "raw" / "rental_price.csv")

        checkpoint_dir = project / "checkpoints" / "rental_price"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_rental_price import RentalPriceTrainer
            trainer = RentalPriceTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        assert (checkpoint_dir / "model.pkl").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "rental_price"
        assert metadata["model_type"] == "lightgbm"
        assert "test_rmse" in metadata["metrics"]
        assert "test_r2" in metadata["metrics"]

        assert (results_dir / "rental_price_metrics.csv").exists()
        assert "test_rmse" in metrics
        assert "test_r2" in metrics

    def test_synthetic_generator_shape(self):
        """Target range and column set match the spec."""
        df = generate_rental_price_data(n_samples=500, seed=0)
        expected_cols = {
            "bedrooms", "bathrooms", "square_feet", "property_type",
            "location_tier", "distance_to_downtown_km", "amenity_score",
            "peer_nightly_rate", "nightly_rate",
        }
        assert set(df.columns) == expected_cols
        assert df["nightly_rate"].between(50, 800).all()
        assert df["bedrooms"].between(0, 6).all()
        assert df["bathrooms"].between(1, 4).all()
        assert df["location_tier"].between(1, 5).all()
