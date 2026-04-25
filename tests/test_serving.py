"""Tests for model serving (uses tiny model fixtures from conftest)."""

import json

from fastapi.testclient import TestClient


def test_credit_risk_prediction(predictor):
    result = predictor.predict_credit_risk({
        "age": 35, "annual_income": 65000, "credit_score": 700,
        "num_open_accounts": 3, "payment_history_pct": 85,
        "debt_to_income_ratio": 0.3, "employment_years": 8, "loan_amount": 25000,
    })
    assert "risk_score" in result
    assert "recommendation" in result
    assert result["recommendation"] in ("APPROVE", "REVIEW", "DECLINE")
    assert 0 <= result["risk_score"] <= 1


def test_fraud_prediction(predictor):
    result = predictor.predict_fraud({
        "transaction_amount": 150, "merchant_category": "online_retail",
        "hour_of_day": 14, "day_of_week": 2, "distance_from_home": 15,
        "is_online": 1, "card_age_days": 365,
        "num_transactions_last_hour": 1, "amount_vs_avg_ratio": 3,
    })
    assert "risk_level" in result
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "reconstruction_error" in result


def test_price_prediction(predictor):
    result = predictor.predict_price({
        "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
        "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
        "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
    })
    assert "predicted_price" in result
    assert result["predicted_price"] > 0
    assert result["price_range_low"] < result["predicted_price"] < result["price_range_high"]


def test_demand_prediction(predictor):
    result = predictor.predict_demand("electronics")
    assert "predictions" in result
    assert len(result["predictions"]) == 7
    assert result["avg_predicted_demand"] > 0


# ---------------------------------------------------------------------------
# A.9.1 — dynamic checkpoint discovery for /models and /health
# ---------------------------------------------------------------------------


def test_get_models_scans_checkpoints_dir(tmp_path):
    """get_model_info() should return one entry per checkpoints/<problem>/metadata.json,
    with no hardcoded list of problem names."""
    from src.serving.predictor import ModelPredictor

    # Drop a fake checkpoint directory with a non-canonical problem name —
    # the dynamic scan must pick this up despite "foo" not being in any list.
    foo_dir = tmp_path / "checkpoints" / "foo"
    foo_dir.mkdir(parents=True)
    (foo_dir / "metadata.json").write_text(
        json.dumps({"problem": "foo", "model_type": "xgboost",
                    "metrics": {"test_auc_roc": 0.91}})
    )

    p = ModelPredictor()
    p.root = tmp_path

    info = p.get_model_info()
    assert "foo" in info, f"Expected dynamic scan to pick up 'foo', got: {list(info)}"
    assert info["foo"]["problem"] == "foo"
    assert info["foo"]["model_type"] == "xgboost"


def test_health_reports_all_existing_checkpoints(tmp_path):
    """GET /health should self-discover every checkpoints/<problem>/ that exists,
    reporting {"available": True} for each. No hardcoded list."""
    import src.serving.api as api_module
    from src.serving.predictor import ModelPredictor

    # Two checkpoint dirs that aren't in any historical hardcoded list
    for name in ("foo", "bar"):
        d = tmp_path / "checkpoints" / name
        d.mkdir(parents=True)
        (d / "metadata.json").write_text(json.dumps({"problem": name}))

    p = ModelPredictor()
    p.root = tmp_path

    original = api_module.predictor
    api_module.predictor = p
    try:
        client = TestClient(api_module.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "models" in body, f"Expected /health to include 'models', got: {body}"
        assert "foo" in body["models"]
        assert "bar" in body["models"]
        assert body["models"]["foo"]["available"] is True
        assert body["models"]["bar"]["available"] is True
    finally:
        api_module.predictor = original


def test_health_reports_empty_models_when_no_checkpoints(tmp_path):
    """When checkpoints/ is empty (or missing), /health must not crash and
    must return models == {}."""
    import src.serving.api as api_module
    from src.serving.predictor import ModelPredictor

    # tmp_path has no checkpoints/ subdir at all
    p = ModelPredictor()
    p.root = tmp_path

    original = api_module.predictor
    api_module.predictor = p
    try:
        client = TestClient(api_module.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["models"] == {}

        # /models must also return {} and not crash
        resp_models = client.get("/models")
        assert resp_models.status_code == 200
        assert resp_models.json() == {}
    finally:
        api_module.predictor = original
