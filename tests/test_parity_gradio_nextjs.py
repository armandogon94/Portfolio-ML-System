"""A.9.10 — Gradio ↔ Next.js parity gate.

Captures + asserts JSON snapshots of /predict/{fraud,price,demand}
for the legacy default inputs. Both Gradio (retiring in A.9.11) and
the Next.js ports (A.9.4–A.9.6) call the same FastAPI endpoints, so
parity at the API level proves the UI swap is functionally
equivalent — what the user sees on /fintech/fraud,
/real-estate/price, and /logistics/demand matches what they would
have seen on the old Gradio tabs for the same inputs.

The default payloads below mirror the Pydantic field defaults in
``src/serving/api.py`` (``Transaction``, ``Property``,
``DemandRequest``), which in turn matched the Gradio tab inputs at
``v1.3.0-phase-a-fanout``. Re-generate the snapshot files when
models are retrained and the new behavior is intentional.

Usage:
    # Verify parity (default; requires real on-disk checkpoints):
    uv run pytest -m parity tests/test_parity_gradio_nextjs.py -v

    # Regenerate snapshots after a model retrain:
    PARITY_SNAPSHOT_REGENERATE=1 uv run pytest -m parity \\
        tests/test_parity_gradio_nextjs.py -v

The test is excluded from ``make test`` via the ``parity`` marker
config in ``pyproject.toml``; run it explicitly before shipping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_module
from src.config import get_project_root
from src.serving.predictor import ModelPredictor

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gradio_parity"
PARITY_SNAPSHOT_REGENERATE = os.environ.get("PARITY_SNAPSHOT_REGENERATE") == "1"

# Tolerance for float comparisons. Generous enough to absorb MPS/CPU
# device drift on Apple Silicon (~1e-5 typical) but tight enough to
# catch a real regression (e.g., a model swap or feature-engineering
# bug would shift values by ≥1e-2).
NUMERIC_TOLERANCE = 1e-3

# Default payloads taken verbatim from the Pydantic field defaults
# in src/serving/api.py at v1.3.0-phase-a-fanout. Each one mirrors
# what the corresponding Gradio tab would have submitted on Click
# without user edits.
DEFAULT_PAYLOADS: dict[str, dict[str, Any]] = {
    "fraud": {
        "endpoint": "/predict/fraud",
        "payload": {
            "transaction_amount": 150.0,
            "merchant_category": "online_retail",
            "hour_of_day": 14,
            "day_of_week": 2,
            "distance_from_home": 15.0,
            "is_online": 1,
            "card_age_days": 365,
            "num_transactions_last_hour": 1,
            "amount_vs_avg_ratio": 3.0,
        },
    },
    "price": {
        "endpoint": "/predict/price",
        "payload": {
            "square_feet": 1800,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2000,
            "lot_size_sqft": 8000,
            "garage_spaces": 2,
            "has_pool": 0,
            "neighborhood_tier": 3,
            "proximity_to_city_center": 10.0,
        },
    },
    "demand": {
        "endpoint": "/predict/demand",
        "payload": {"product": "electronics"},
    },
}

REQUIRED_CHECKPOINTS = (
    "credit_risk",
    "fraud_detection",
    "price_prediction",
    "demand_forecasting",
)


def _real_checkpoints_present() -> bool:
    root = get_project_root() / "checkpoints"
    return all((root / name / "metadata.json").exists() for name in REQUIRED_CHECKPOINTS)


@pytest.fixture(scope="module")
def real_api_client():
    """FastAPI TestClient bound to the real on-disk checkpoints.

    Distinct from ``conftest.py::api_client``, which uses tiny test
    fixtures unsuitable for parity (output values would drift wildly
    from the trained models). This fixture rebinds
    ``src.serving.api.predictor`` to a fresh ``ModelPredictor()``
    pointed at the project root for the duration of the module,
    then restores the original predictor afterwards.
    """
    if not _real_checkpoints_present():
        pytest.skip(
            "Real checkpoints not present — run `make all` first or skip the parity test."
        )

    original = api_module.predictor
    api_module.predictor = ModelPredictor()
    client = TestClient(api_module.app)
    try:
        yield client
    finally:
        api_module.predictor = original


def _compare(actual: Any, expected: Any, path: str = "") -> None:
    """Recursive equality check with float tolerance.

    - dicts compare by exact key set (so an unexpected field is a
      regression worth flagging)
    - lists compare element-wise (and length-wise)
    - floats compare within NUMERIC_TOLERANCE
    - everything else compares by ``==``
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{path}: key mismatch — "
            f"actual={sorted(actual.keys())} expected={sorted(expected.keys())}"
        )
        for k in expected:
            _compare(actual[k], expected[k], f"{path}.{k}" if path else k)
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list"
        assert len(actual) == len(expected), (
            f"{path}: length mismatch — actual={len(actual)} expected={len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)), (
            f"{path}: expected number, got {type(actual).__name__}"
        )
        assert abs(actual - expected) <= NUMERIC_TOLERANCE, (
            f"{path}: {actual} vs {expected} exceeds tolerance {NUMERIC_TOLERANCE}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.mark.parity
@pytest.mark.parametrize("name", sorted(DEFAULT_PAYLOADS.keys()))
def test_predict_endpoint_matches_committed_snapshot(
    name: str, real_api_client: TestClient
) -> None:
    """Each /predict/{model} response matches its v1.3.0 snapshot."""
    spec = DEFAULT_PAYLOADS[name]
    response = real_api_client.post(spec["endpoint"], json=spec["payload"])
    assert response.status_code == 200, (
        f"{spec['endpoint']} returned {response.status_code}: {response.text}"
    )
    actual = response.json()

    snapshot_path = FIXTURE_DIR / f"{name}.json"

    if PARITY_SNAPSHOT_REGENERATE or not snapshot_path.exists():
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        # In regenerate mode, succeed without comparing — the next
        # default-mode run is what proves stability.
        if PARITY_SNAPSHOT_REGENERATE:
            pytest.skip(f"Regenerated {snapshot_path.relative_to(snapshot_path.parents[2])}")
        # First-time capture (no env flag): also skip, instructing
        # the user to commit the snapshot.
        pytest.skip(
            f"Captured initial snapshot at {snapshot_path}. Commit and re-run."
        )

    expected = json.loads(snapshot_path.read_text())
    _compare(actual, expected)
