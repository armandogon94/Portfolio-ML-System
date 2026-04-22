"""Tests for scripts/train.py CLI — Phase A.1.9 --modality flag."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import get_project_root
from src.data.generate_housing import generate_housing_data

# Make scripts/ importable
sys.path.insert(0, str(Path(get_project_root()) / "scripts"))


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    shutil.copytree(real_root / "configs", tmp_path / "configs")
    return tmp_path


def _tiny_zillow_csv(path: Path) -> None:
    """Minimal Zillow-shaped CSV that the housing_adapter can map."""
    import numpy as np

    rng = np.random.default_rng(11)
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
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _mock_kagglehub(monkeypatch, zillow_dir: Path) -> None:
    """Install a stub kagglehub that returns our tiny Zillow dir."""
    monkeypatch.setitem(
        sys.modules, "kagglehub",
        SimpleNamespace(dataset_download=lambda slug: str(zillow_dir)),
    )
    monkeypatch.setattr(
        "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
    )


def _patch_price_for_tiny_run(monkeypatch) -> None:
    """Force PricePredictionTrainer to use tiny n_estimators for speed."""
    from src.training import train_price as tp
    orig_init = tp.PricePredictionTrainer.__init__

    def tiny_init(self, use_wandb=True, modality=None):
        orig_init(self, use_wandb=use_wandb, modality=modality)
        self.config["model"]["params"]["n_estimators"] = 3

    monkeypatch.setattr(tp.PricePredictionTrainer, "__init__", tiny_init)


# ---------------------------------------------------------------------------
# --modality single value
# ---------------------------------------------------------------------------


class TestCliSingleModality:

    def test_cli_modality_synthetic_writes_single_checkpoint(
        self, tmp_path, monkeypatch
    ):
        """--modality synthetic trains once, writes _synthetic checkpoint + legacy mirror."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
        df.to_csv(project / "data" / "raw" / "housing.csv", index=False)

        _patch_price_for_tiny_run(monkeypatch)

        with patch("src.config.PROJECT_ROOT", project):
            import scripts.train as train_mod
            monkeypatch.setattr(
                sys, "argv",
                ["train.py", "--model", "price", "--modality", "synthetic", "--no-wandb"],
            )
            train_mod.main()

        # Modality-suffixed checkpoint exists
        assert (project / "checkpoints" / "price_prediction_synthetic" / "model.pkl").exists()
        # Legacy mirror exists (from A.1.8)
        assert (project / "checkpoints" / "price_prediction" / "model.pkl").exists()
        # Modality CSV under results/modalities/
        assert (
            project / "results" / "modalities" / "price_prediction_synthetic.csv"
        ).exists()
        # Legacy metrics CSV mirrored
        assert (project / "results" / "price_prediction_metrics.csv").exists()


# ---------------------------------------------------------------------------
# --modality all
# ---------------------------------------------------------------------------


class TestCliModalityAll:

    def test_cli_modality_all_trains_all_three_and_writes_comparison(
        self, tmp_path, monkeypatch, caplog
    ):
        """--modality all produces 3 checkpoints, 3 modality CSVs, 1 comparison CSV."""
        import logging

        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
        df.to_csv(project / "data" / "raw" / "housing.csv", index=False)

        zillow_dir = tmp_path / "zillow"
        zillow_dir.mkdir()
        _tiny_zillow_csv(zillow_dir / "zillow_sales.csv")
        _mock_kagglehub(monkeypatch, zillow_dir)
        _patch_price_for_tiny_run(monkeypatch)

        with caplog.at_level(logging.INFO), \
             patch("src.config.PROJECT_ROOT", project):
            import scripts.train as train_mod
            monkeypatch.setattr(
                sys, "argv",
                ["train.py", "--model", "price", "--modality", "all", "--no-wandb"],
            )
            train_mod.main()

        # 3 modality checkpoints
        for modality in ("synthetic", "stream", "mixed"):
            ckpt = project / "checkpoints" / f"price_prediction_{modality}"
            assert (ckpt / "model.pkl").exists(), f"missing {modality} checkpoint"
            assert (ckpt / "metadata.json").exists(), f"missing {modality} metadata"

        # Comparison CSV with 3 rows, modality + recommended columns
        comparison_path = (
            project / "results" / "modality_comparison_price_prediction.csv"
        )
        assert comparison_path.exists()
        comparison = pd.read_csv(comparison_path)
        assert len(comparison) == 3
        assert set(comparison["modality"]) == {"synthetic", "stream", "mixed"}
        assert "recommended" in comparison.columns
        # Exactly one modality is recommended
        assert comparison["recommended"].sum() == 1

        # The recommended modality's metadata.json has recommended=True
        recommended_modality = comparison.loc[
            comparison["recommended"], "modality"
        ].iloc[0]
        with open(
            project / "checkpoints" / f"price_prediction_{recommended_modality}" /
            "metadata.json"
        ) as f:
            metadata = json.load(f)
        assert metadata.get("recommended") is True

        # Other modalities' metadata do NOT claim recommended
        for other in {"synthetic", "stream", "mixed"} - {recommended_modality}:
            with open(
                project / "checkpoints" / f"price_prediction_{other}" /
                "metadata.json"
            ) as f:
                other_metadata = json.load(f)
            assert other_metadata.get("recommended") is not True


# ---------------------------------------------------------------------------
# --modality on non-modality models
# ---------------------------------------------------------------------------


class TestCliModalityBackwardCompat:

    def test_cli_no_modality_uses_legacy_path(self, tmp_path, monkeypatch):
        """Without --modality, trainer runs legacy-style (no _<modality>/ suffix)."""
        project = _setup_tmp_project(tmp_path)
        df = generate_housing_data(n_samples=200, seed=42)
        (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
        df.to_csv(project / "data" / "raw" / "housing.csv", index=False)

        _patch_price_for_tiny_run(monkeypatch)

        with patch("src.config.PROJECT_ROOT", project):
            import scripts.train as train_mod
            monkeypatch.setattr(
                sys, "argv",
                ["train.py", "--model", "price", "--no-wandb"],
            )
            train_mod.main()

        # Legacy path used
        assert (project / "checkpoints" / "price_prediction" / "model.pkl").exists()
        # No modality suffix
        assert not (
            project / "checkpoints" / "price_prediction_synthetic"
        ).exists()
        # Legacy CSV, no modality dir
        assert (project / "results" / "price_prediction_metrics.csv").exists()
        assert not (project / "results" / "modalities").exists()


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """End any dangling MLflow runs between tests."""
    import mlflow
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
