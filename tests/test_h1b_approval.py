"""H-1B approval pipeline integration tests.

Mirrors the shape of ``tests/test_training_pipeline.py::TestCreditRiskPipeline``
— generate tiny synthetic data, run the trainer end-to-end on an isolated
project root, and assert checkpoint + metrics + predictor behavior.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import get_project_root
from src.data.generate_h1b_approval import generate_h1b_approval_data


def _setup_tmp_project(tmp_path: Path) -> Path:
    """Copy configs into tmp_path and return it as a fake project root."""
    real_root = get_project_root()
    configs_dest = tmp_path / "configs"
    shutil.copytree(real_root / "configs", configs_dest)
    return tmp_path


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestGenerateH1BData:
    """Synthetic generator sanity checks."""

    def test_expected_columns(self):
        df = generate_h1b_approval_data(n_samples=200, seed=42)
        expected = {
            "prevailing_wage",
            "soc_code_level",
            "employer_size_tier",
            "job_level",
            "education_level",
            "experience_years",
            "country_of_citizenship_tier",
            "employer_prior_approval_rate",
            "is_approved",
        }
        assert set(df.columns) == expected
        assert len(df) == 200

    def test_feature_ranges(self):
        """All features must lie in the documented ranges."""
        df = generate_h1b_approval_data(n_samples=500, seed=42)

        assert df["prevailing_wage"].between(40_000, 300_000).all()
        assert df["soc_code_level"].between(1, 4).all()
        assert df["employer_size_tier"].between(1, 5).all()
        assert df["job_level"].between(1, 4).all()
        assert df["education_level"].between(1, 5).all()
        assert df["experience_years"].between(0, 30).all()
        assert df["country_of_citizenship_tier"].between(1, 5).all()
        assert df["employer_prior_approval_rate"].between(0, 1).all()
        assert df["is_approved"].isin([0, 1]).all()

    def test_approval_rate_in_expected_band(self):
        """Approval rate should land near the 70% target (60–85% band).

        Band is intentionally wide — the logistic + noise combination
        drifts a few percentage points between seeds.
        """
        df = generate_h1b_approval_data(n_samples=10_000, seed=42)
        approval_rate = df["is_approved"].mean()
        assert 0.60 <= approval_rate <= 0.85, (
            f"Approval rate {approval_rate:.2%} outside 60–85% band"
        )


class TestH1BApprovalPipeline:
    """Full trainer pipeline on tiny data under an isolated project root."""

    def test_full_pipeline(self, tmp_path):
        project = _setup_tmp_project(tmp_path)
        df = generate_h1b_approval_data(n_samples=500, seed=42)
        _write_csv(df, project / "data" / "raw" / "h1b_approval.csv")

        checkpoint_dir = project / "checkpoints" / "h1b_approval"
        results_dir = project / "results"

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_h1b_approval import H1BApprovalTrainer

            trainer = H1BApprovalTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 5
            metrics = trainer.run()

        # Checkpoint artifacts
        assert (checkpoint_dir / "model.json").exists()
        assert (checkpoint_dir / "metadata.json").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["problem"] == "h1b_approval"
        assert metadata["model_type"] == "xgboost"
        assert "test_auc_roc" in metadata["metrics"]

        # Results CSV
        assert (results_dir / "h1b_approval_metrics.csv").exists()
        assert "test_accuracy" in metrics
        assert "test_auc_roc" in metrics


class TestH1BPredictor:
    """ModelPredictor.predict_h1b_approval exercises the serving path end-to-end."""

    @pytest.fixture(scope="class")
    def trained_predictor(self, tmp_path_factory):
        """Train the model once for the whole class, mount a predictor against it.

        Uses a moderate dataset size (5k rows, 50 trees) so the predictor
        separates the strong/weak end cases with high confidence — tiny
        fits can collapse into a majority-class predictor and spoil the
        end-to-end sanity checks below.
        """
        tmp_path = tmp_path_factory.mktemp("h1b_predictor")
        project = _setup_tmp_project(tmp_path)
        df = generate_h1b_approval_data(n_samples=5000, seed=42)
        _write_csv(df, project / "data" / "raw" / "h1b_approval.csv")

        with patch("src.config.PROJECT_ROOT", project):
            from src.training.train_h1b_approval import H1BApprovalTrainer

            trainer = H1BApprovalTrainer(use_wandb=False)
            trainer.config["model"]["params"]["n_estimators"] = 50
            trainer.run()

            # The predictor reads PROJECT_ROOT at construction; stay inside the patch.
            from src.serving.predictor import ModelPredictor

            predictor = ModelPredictor()
            yield predictor

    def test_predict_returns_expected_shape(self, trained_predictor):
        result = trained_predictor.predict_h1b_approval(
            {
                "prevailing_wage": 120_000,
                "soc_code_level": 3,
                "employer_size_tier": 3,
                "job_level": 2,
                "education_level": 2,
                "experience_years": 5,
                "country_of_citizenship_tier": 2,
                "employer_prior_approval_rate": 0.75,
            }
        )
        assert set(result) == {"probability_approval", "recommendation", "confidence"}
        assert 0.0 <= result["probability_approval"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["recommendation"] in {
            "APPROVE_LIKELY",
            "REVIEW",
            "DECLINE_LIKELY",
        }

    def test_strong_case_approves(self, trained_predictor):
        """Very high wage + top employer + PhD ⇒ APPROVE_LIKELY."""
        result = trained_predictor.predict_h1b_approval(
            {
                "prevailing_wage": 200_000,
                "soc_code_level": 4,
                "employer_size_tier": 5,
                "job_level": 4,
                "education_level": 3,
                "experience_years": 15,
                "country_of_citizenship_tier": 1,
                "employer_prior_approval_rate": 0.95,
            }
        )
        assert result["recommendation"] == "APPROVE_LIKELY"
        assert result["probability_approval"] >= 0.7

    def test_weak_case_declines(self, trained_predictor):
        """Low wage + tiny employer + weak track record ⇒ DECLINE_LIKELY."""
        result = trained_predictor.predict_h1b_approval(
            {
                "prevailing_wage": 45_000,
                "soc_code_level": 1,
                "employer_size_tier": 1,
                "job_level": 1,
                "education_level": 1,
                "experience_years": 0,
                "country_of_citizenship_tier": 5,
                "employer_prior_approval_rate": 0.15,
            }
        )
        assert result["recommendation"] == "DECLINE_LIKELY"
        assert result["probability_approval"] < 0.4

    def test_explain_returns_shap(self, trained_predictor):
        """explain_h1b_approval returns SHAP feature importances."""
        result = trained_predictor.explain_h1b_approval(
            {
                "prevailing_wage": 120_000,
                "soc_code_level": 3,
                "employer_size_tier": 3,
                "job_level": 2,
                "education_level": 2,
                "experience_years": 5,
                "country_of_citizenship_tier": 2,
                "employer_prior_approval_rate": 0.75,
            }
        )
        assert result["explanation_type"] == "shap"
        assert "feature_importances" in result
        assert "top_features" in result
        # Every feature the model saw should be in the importance map.
        assert "prevailing_wage" in result["feature_importances"]
        assert "employer_prior_approval_rate" in result["feature_importances"]
