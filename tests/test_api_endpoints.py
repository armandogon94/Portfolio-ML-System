"""API integration tests using FastAPI TestClient with tiny model fixtures."""

import pytest


class TestHealthEndpoint:

    def test_health_returns_ok(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelsEndpoint:

    def test_models_returns_dict(self, api_client):
        resp = api_client.get("/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


class TestCreditRiskEndpoint:

    def test_predict_with_defaults(self, api_client):
        resp = api_client.post("/predict/credit-risk", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "recommendation" in data
        assert "confidence" in data

    def test_predict_valid_response_shape(self, api_client):
        resp = api_client.post("/predict/credit-risk", json={
            "age": 35, "annual_income": 65000, "credit_score": 700,
            "num_open_accounts": 3, "payment_history_pct": 85.0,
            "debt_to_income_ratio": 0.3, "employment_years": 8.0,
            "loan_amount": 25000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["risk_score"] <= 1
        assert data["recommendation"] in ("APPROVE", "REVIEW", "DECLINE")
        assert 0 <= data["confidence"] <= 1

    @pytest.mark.parametrize("field,value", [
        ("age", 25), ("annual_income", 120000), ("credit_score", 800),
        ("loan_amount", 5000),
    ])
    def test_predict_with_single_field(self, api_client, field, value):
        resp = api_client.post("/predict/credit-risk", json={field: value})
        assert resp.status_code == 200
        assert "risk_score" in resp.json()


class TestFraudEndpoint:

    def test_predict_with_defaults(self, api_client):
        resp = api_client.post("/predict/fraud", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_level" in data
        assert "fraud_probability" in data
        assert "reconstruction_error" in data

    def test_predict_valid_response_shape(self, api_client):
        resp = api_client.post("/predict/fraud", json={
            "transaction_amount": 150.0, "merchant_category": "online_retail",
            "hour_of_day": 14, "day_of_week": 2, "distance_from_home": 15.0,
            "is_online": 1, "card_age_days": 365,
            "num_transactions_last_hour": 1, "amount_vs_avg_ratio": 3.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert "is_anomaly_autoencoder" in data
        assert "is_anomaly_isolation_forest" in data


class TestPriceEndpoint:

    def test_predict_with_defaults(self, api_client):
        resp = api_client.post("/predict/price", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_price" in data
        assert "price_range_low" in data
        assert "price_range_high" in data

    def test_predict_price_positive(self, api_client):
        resp = api_client.post("/predict/price", json={
            "square_feet": 2000, "bedrooms": 3, "bathrooms": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["price_range_low"] < data["predicted_price"] < data["price_range_high"]


class TestDemandEndpoint:

    def test_predict_with_default_product(self, api_client):
        resp = api_client.post("/predict/demand", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 7
        assert data["forecast_days"] == 7

    def test_predict_returns_product_name(self, api_client):
        resp = api_client.post("/predict/demand", json={"product": "electronics"})
        assert resp.status_code == 200
        assert resp.json()["product"] == "electronics"
