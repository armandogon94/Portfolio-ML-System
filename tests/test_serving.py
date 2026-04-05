"""Tests for model serving (requires trained checkpoints)."""

import pytest

# Import xgboost before torch to avoid libomp conflict
import xgboost  # noqa: F401
import lightgbm  # noqa: F401

from src.serving.predictor import ModelPredictor


@pytest.fixture(scope="module")
def predictor():
    return ModelPredictor()


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
