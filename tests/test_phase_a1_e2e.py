"""Phase A.1 end-to-end contract test — one test exercises the whole streaming stack.

This is the guard test for Phase A.1. It verifies every promise the
SPEC.md §"Phase A.1" made, running the full `scripts.train` CLI with
--modality all against tiny mocked data:

1. **Zero network**: kagglehub is mocked; the mock must be called, proving
   the real network path was never taken. W&B is disabled via --no-wandb.
   MLflow goes to local file store (mlruns/), not a remote server.
2. **Three checkpoints**: checkpoints/price_prediction_{synthetic,stream,mixed}/
   each contain model.pkl + metadata.json with the right modality field.
3. **Legacy mirror**: checkpoints/price_prediction/ (the ModelPredictor path)
   is bytes-identical to the synthetic checkpoint.
4. **Modality CSVs**: results/modalities/price_prediction_{m}.csv for each m.
5. **Legacy results mirror**: results/price_prediction_metrics.csv mirrors
   the synthetic modality's metrics.
6. **Comparison CSV**: results/modality_comparison_price_prediction.csv has
   exactly 3 rows (one per modality), a `modality` column, a `recommended`
   column with exactly one True, and numeric metric columns.
7. **Recommended consistency**: only the flagged modality's metadata.json
   has `recommended: True`; the other two either lack the key or are False.
8. **Timing**: test completes within a generous budget on tiny data.
9. **data/raw/**: no Kaggle data leaked into the repo — external cache
   lives in the tmp fixture, not the project tree.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import get_project_root
from src.data.generate_housing import generate_housing_data

# Make scripts/ importable
sys.path.insert(0, str(Path(get_project_root()) / "scripts"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Copy configs into a throwaway project root so tests don't pollute the repo."""
    real_root = get_project_root()
    shutil.copytree(real_root / "configs", tmp_path / "configs")
    # Synthetic housing data ready at the canonical path
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    df = generate_housing_data(n_samples=200, seed=42)
    df.to_csv(tmp_path / "data" / "raw" / "housing.csv", index=False)
    return tmp_path


@pytest.fixture
def tiny_zillow_cache(tmp_path):
    """Minimal Zillow-shaped CSV the housing_adapter can consume."""
    import numpy as np

    zillow_dir = tmp_path / "zillow_cache"
    zillow_dir.mkdir()
    rng = np.random.default_rng(13)
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
    df.to_csv(zillow_dir / "zillow_sales.csv", index=False)
    return zillow_dir


def _reset_mlflow_internal_state() -> None:
    """Clear MLflow module-level caches that leak between tests.

    MLflow caches (a) the active-run stack and (b) the active experiment ID.
    When a prior test's tracking dir gets torn down, these refs dangle and
    crash the next ``start_run``/``set_experiment`` call. This helper clears
    both so the test can establish a fresh tracking context.
    """
    from mlflow.tracking import fluent

    # Active-run stack: clear either the modern contextvar list or legacy list.
    for _ in range(10):
        try:
            stack = fluent._active_run_stack.get()
            if not stack:
                break
            stack.clear()
        except Exception:
            try:
                fluent._active_run_stack.clear()
            except Exception:
                pass
            break

    # Active experiment ID — force re-resolution on next set_experiment call.
    try:
        fluent._active_experiment_id = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Reset MLflow's internal state before and after each test in this file."""
    import mlflow

    _reset_mlflow_internal_state()
    yield
    # Try to end any real active runs gracefully before the final reset.
    for _ in range(10):
        try:
            if mlflow.active_run() is None:
                break
            mlflow.end_run()
        except Exception:
            break
    _reset_mlflow_internal_state()


# ---------------------------------------------------------------------------
# The contract test
# ---------------------------------------------------------------------------


class TestPhaseA1EndToEnd:

    def test_modality_all_honors_full_a1_contract(
        self, tmp_project, tiny_zillow_cache, monkeypatch
    ):
        """One test, nine contract assertions, zero network calls."""
        # ─── Arrange ────────────────────────────────────────────────────

        # Count kagglehub invocations so we can prove the mock was hit
        # (i.e., the real network path was never taken).
        kagglehub_calls: list[str] = []

        fake_kagglehub = SimpleNamespace(
            dataset_download=lambda slug: (
                kagglehub_calls.append(slug),
                str(tiny_zillow_cache),
            )[1],
        )
        monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
        monkeypatch.setattr(
            "src.data.kaggle_credentials.ensure_kaggle_env", lambda: None
        )

        # Force tiny LightGBM run (n_estimators=3) for speed
        from src.training import train_price as tp
        orig_init = tp.PricePredictionTrainer.__init__

        def tiny_init(self, use_wandb=True, modality=None):
            orig_init(self, use_wandb=use_wandb, modality=modality)
            self.config["model"]["params"]["n_estimators"] = 3

        monkeypatch.setattr(tp.PricePredictionTrainer, "__init__", tiny_init)

        # Record pre-run data/raw/ size so we can prove we didn't bloat it
        raw_dir = tmp_project / "data" / "raw"
        size_before = sum(p.stat().st_size for p in raw_dir.rglob("*") if p.is_file())

        # Isolate MLflow to a tmp tracking dir to avoid stale run refs
        # leaked by prior tests in the suite. Setting the env var alone is
        # NOT sufficient — MLflow caches the tracking URI at module level,
        # so we also call set_tracking_uri explicitly (matches the pattern
        # in tests/test_mlflow_integration.py). Also clear any inherited
        # MLFLOW_EXPERIMENT_ID / _NAME — prior tests may have set these to
        # IDs that don't exist in our fresh tracking dir.
        import mlflow as _mlflow
        tracking_uri = str(tmp_project / "mlruns")
        monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
        _mlflow.set_tracking_uri(tracking_uri)

        # ─── Act ────────────────────────────────────────────────────────
        start = time.perf_counter()
        with patch("src.config.PROJECT_ROOT", tmp_project):
            import scripts.train as train_mod
            monkeypatch.setattr(
                sys, "argv",
                ["train.py", "--model", "price", "--modality", "all", "--no-wandb"],
            )
            train_mod.main()
        elapsed = time.perf_counter() - start

        # ─── Assertions ─────────────────────────────────────────────────

        # (1) ZERO NETWORK — the fake kagglehub was invoked, not the real one.
        #     mixed and stream both trigger a Kaggle fetch → 2 calls expected.
        assert len(kagglehub_calls) == 2, (
            f"Expected 2 kagglehub calls (stream + mixed), got {kagglehub_calls}"
        )
        assert all(
            slug == "computingvictor/zillow-market-analysis-and-real-estate-sales-data"
            for slug in kagglehub_calls
        ), kagglehub_calls

        # (2) THREE CHECKPOINTS with correct modality field
        for modality in ("synthetic", "stream", "mixed"):
            ckpt = tmp_project / "checkpoints" / f"price_prediction_{modality}"
            assert (ckpt / "model.pkl").exists(), f"missing {modality}/model.pkl"
            assert (ckpt / "metadata.json").exists(), f"missing {modality}/metadata.json"
            with open(ckpt / "metadata.json") as f:
                metadata = json.load(f)
            assert metadata["modality"] == modality, (
                f"{modality} metadata.json has modality={metadata['modality']}"
            )

        # (3) LEGACY CHECKPOINT MIRROR — synthetic replicated to
        #     checkpoints/price_prediction/ for ModelPredictor compat.
        synthetic_bytes = (
            tmp_project / "checkpoints" / "price_prediction_synthetic" / "model.pkl"
        ).read_bytes()
        legacy_bytes = (
            tmp_project / "checkpoints" / "price_prediction" / "model.pkl"
        ).read_bytes()
        assert synthetic_bytes == legacy_bytes, "legacy checkpoint not a bytes-exact mirror"

        # (4) MODALITY CSVs per modality
        for modality in ("synthetic", "stream", "mixed"):
            csv = tmp_project / "results" / "modalities" / f"price_prediction_{modality}.csv"
            assert csv.exists(), f"missing modality CSV for {modality}"
            df = pd.read_csv(csv)
            assert set(df.columns) == {"metric", "value"}, df.columns.tolist()
            assert "test_r2" in set(df["metric"]), f"{modality} CSV missing test_r2"

        # (5) LEGACY RESULTS CSV MIRROR — synthetic metrics mirrored.
        legacy_csv = tmp_project / "results" / "price_prediction_metrics.csv"
        assert legacy_csv.exists()
        synthetic_csv = tmp_project / "results" / "modalities" / "price_prediction_synthetic.csv"
        assert pd.read_csv(legacy_csv).equals(pd.read_csv(synthetic_csv))

        # (6) COMPARISON CSV shape & content
        comparison_path = (
            tmp_project / "results" / "modality_comparison_price_prediction.csv"
        )
        assert comparison_path.exists()
        comparison = pd.read_csv(comparison_path)
        assert len(comparison) == 3, f"comparison has {len(comparison)} rows"
        assert set(comparison["modality"]) == {"synthetic", "stream", "mixed"}
        assert "recommended" in comparison.columns
        # Must contain at least the key metric (test_r2) as a numeric column
        assert "test_r2" in comparison.columns, comparison.columns.tolist()
        # Exactly one row flagged recommended
        assert comparison["recommended"].sum() == 1

        # (7) RECOMMENDED CONSISTENCY — only the winner's metadata says True.
        recommended_modality = comparison.loc[
            comparison["recommended"], "modality"
        ].iloc[0]
        for modality in ("synthetic", "stream", "mixed"):
            meta_path = (
                tmp_project / "checkpoints" / f"price_prediction_{modality}" /
                "metadata.json"
            )
            with open(meta_path) as f:
                metadata = json.load(f)
            if modality == recommended_modality:
                assert metadata.get("recommended") is True
            else:
                assert metadata.get("recommended") is not True, (
                    f"{modality} should not be flagged recommended"
                )

        # (8) TIMING — tiny data + tiny n_estimators should finish fast.
        #     Generous budget (30s) accounts for CI variance; expected ~3-5s.
        assert elapsed < 30.0, f"E2E took {elapsed:.1f}s (budget 30s)"

        # (9) NO REPO BLOAT — data/raw/ unchanged; external data lives outside.
        size_after = sum(p.stat().st_size for p in raw_dir.rglob("*") if p.is_file())
        assert size_after == size_before, (
            f"data/raw/ grew by {size_after - size_before} bytes — "
            "external data should never be written into the project tree"
        )
