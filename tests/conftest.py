"""Shared fixtures. Everything here runs on the committed CI fixtures.

Two environment facts set before any import, both load-bearing on macOS:

* ``KMP_DUPLICATE_LIB_OK``: XGBoost, LightGBM and PyTorch each vendor their own
  libomp. Importing two of them in one process aborts without this.
* ``OMP_NUM_THREADS=1``: deterministic tree building, and it keeps a full test
  run from saturating all 10 cores.

MPS is forced off for the whole session. torch 2.13.0 on this hardware deadlocked a
CPU tensor loop that followed an MPS matmul **in the same process**. A pytest
session is exactly that shape, since one test can touch MPS and the next CPU. The
device-selection logic itself is tested separately with an explicit mock.
"""

from __future__ import annotations

import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import lightgbm  # noqa: F401,E402  - import before torch (libomp)
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xgboost  # noqa: F401,E402  - import before torch (libomp)

_mps_off = patch("torch.backends.mps.is_available", return_value=False)
_mps_off.start()

from src.config import get_project_root, load_config  # noqa: E402
from src.data.adapters import credit_card_churn, ieee_cis, lending_club  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return get_project_root()


@pytest.fixture(scope="session")
def fraud_config() -> dict:
    return load_config("fraud")


@pytest.fixture(scope="session")
def credit_risk_config() -> dict:
    return load_config("credit_risk")


@pytest.fixture(scope="session")
def churn_config() -> dict:
    return load_config("churn")


@pytest.fixture(scope="session")
def fraud_frame():
    """The IEEE-CIS CI fixture, through the adapter."""
    return ieee_cis.load(sample=True)


@pytest.fixture(scope="session")
def credit_risk_frame():
    """The LendingClub CI fixture, through the adapter."""
    return lending_club.load(sample=True)


@pytest.fixture(scope="session")
def churn_frame():
    """The attrition CI fixture, through the adapter."""
    return credit_card_churn.load(sample=True)


@pytest.fixture(scope="session")
def trained_churn(tmp_path_factory):
    """A real trained churn checkpoint on disk, built from the CI fixture.

    Churn is used for the serving and e2e tests because it is the smallest of the
    three (24 columns) and trains in well under a second. The checkpoint is written
    to a tmp ``checkpoints/`` tree so ``CheckpointRegistry`` can discover it exactly
    as it would in production, with no monkeypatching of the loader.

    Returns:
        ``(checkpoints_root, problem_name)``.
    """
    import joblib

    from src.data.split import make_splits
    from src.training.tabular import TabularTrainer

    root = tmp_path_factory.mktemp("registry_root") / "checkpoints"
    directory = root / "churn"
    directory.mkdir(parents=True)

    trainer = TabularTrainer("churn", sample=True)
    frame = trainer.load_data()
    split = make_splits(frame, {"type": "random", "test_size": 0.25}, "is_attrited", 42)[0]
    data = trainer.build_matrix(frame, split)
    model = trainer.train(data)

    joblib.dump(model, directory / "model.joblib")
    joblib.dump(trainer.feature_bundle(), directory / "features.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "problem": "churn",
                "model_type": "lightgbm",
                "seed": 42,
                "git_sha": "0" * 40,
                "dataset": {"slug": "SYNTHETIC-CI-FIXTURE-NOT-REAL-DATA"},
                "feature_columns": trainer.feature_columns,
                "metrics": {"test_roc_auc": 0.5},
                "note": "Built from a CI fixture. Not a published result.",
            }
        )
    )
    trainer.finish()
    return root, "churn"


@pytest.fixture
def registry(trained_churn):
    """A :class:`CheckpointRegistry` pointed at the fixture-built checkpoint."""
    from src.serving.registry import CheckpointRegistry

    root, _ = trained_churn
    return CheckpointRegistry(root)


@pytest.fixture
def api_client(trained_churn, monkeypatch):
    """FastAPI ``TestClient`` whose registry sees the fixture-built checkpoint."""
    from fastapi.testclient import TestClient

    import src.serving.api as api

    root, _ = trained_churn
    from src.serving.registry import CheckpointRegistry

    monkeypatch.setattr(api, "registry", CheckpointRegistry(root))
    return TestClient(api.app)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
