"""Tests for feature engineering pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.features.credit_risk_features import engineer_features as credit_features
from src.features.credit_risk_features import get_feature_columns as credit_cols
from src.features.fraud_features import engineer_features as fraud_features
from src.features.fraud_features import get_feature_columns as fraud_cols
from src.features.housing_features import engineer_features as housing_features
from src.features.housing_features import get_feature_columns as housing_cols
from src.features.timeseries_features import DemandDataset, create_sequences

# ---------------------------------------------------------------------------
# Credit Risk Features
# ---------------------------------------------------------------------------


class TestCreditRiskFeatures:

    def test_all_feature_columns_present(self, credit_risk_df):
        result = credit_features(credit_risk_df)
        for col in credit_cols():
            assert col in result.columns, f"Missing: {col}"

    def test_loan_to_income_value(self):
        df = pd.DataFrame([{
            "age": 35, "annual_income": 100000, "credit_score": 700,
            "num_open_accounts": 3, "payment_history_pct": 85,
            "debt_to_income_ratio": 0.3, "employment_years": 8, "loan_amount": 25000,
        }])
        result = credit_features(df)
        assert result["loan_to_income"].iloc[0] == pytest.approx(0.25)

    def test_credit_utilization_value(self):
        df = pd.DataFrame([{
            "age": 35, "annual_income": 65000, "credit_score": 700,
            "num_open_accounts": 5, "payment_history_pct": 85,
            "debt_to_income_ratio": 0.4, "employment_years": 8, "loan_amount": 25000,
        }])
        result = credit_features(df)
        assert result["credit_utilization"].iloc[0] == pytest.approx(2.0)

    def test_employment_stability_value(self):
        df = pd.DataFrame([{
            "age": 38, "annual_income": 65000, "credit_score": 700,
            "num_open_accounts": 3, "payment_history_pct": 85,
            "debt_to_income_ratio": 0.3, "employment_years": 10, "loan_amount": 25000,
        }])
        result = credit_features(df)
        expected = 10.0 / (38 - 18)
        assert result["employment_stability"].iloc[0] == pytest.approx(expected)

    @pytest.mark.parametrize("score,expected_tier", [
        (500, 0), (620, 1), (700, 2), (770, 3), (820, 4),
    ])
    def test_credit_tier_mapping(self, score, expected_tier):
        df = pd.DataFrame([{
            "age": 35, "annual_income": 65000, "credit_score": score,
            "num_open_accounts": 3, "payment_history_pct": 85,
            "debt_to_income_ratio": 0.3, "employment_years": 8, "loan_amount": 25000,
        }])
        result = credit_features(df)
        assert result["credit_tier"].iloc[0] == expected_tier

    def test_no_nan_in_output(self, credit_risk_df):
        result = credit_features(credit_risk_df)
        feature_data = result[credit_cols()]
        assert feature_data.notna().all().all()

    def test_does_not_mutate_input(self, credit_risk_df):
        original = credit_risk_df.copy()
        credit_features(credit_risk_df)
        pd.testing.assert_frame_equal(credit_risk_df, original)


# ---------------------------------------------------------------------------
# Fraud Features
# ---------------------------------------------------------------------------


class TestFraudFeatures:

    def test_all_feature_columns_present(self, fraud_df):
        result, _ = fraud_features(fraud_df)
        for col in fraud_cols():
            assert col in result.columns, f"Missing: {col}"

    def test_returns_artifacts_dict(self, fraud_df):
        _, artifacts = fraud_features(fraud_df)
        assert "merchant_encoder" in artifacts

    def test_merchant_encoding_numeric(self, fraud_df):
        result, _ = fraud_features(fraud_df)
        assert np.issubdtype(result["merchant_category_encoded"].dtype, np.integer)

    def test_log_amount_computed(self):
        df = pd.DataFrame([{
            "transaction_amount": 100.0, "merchant_category": "grocery",
            "hour_of_day": 14, "day_of_week": 2, "distance_from_home": 10,
            "is_online": 0, "card_age_days": 365,
            "num_transactions_last_hour": 1, "amount_vs_avg_ratio": 2.0,
            "is_fraud": 0,
        }])
        result, _ = fraud_features(df)
        expected = np.log1p(100.0)
        assert result["log_amount"].iloc[0] == pytest.approx(expected)

    @pytest.mark.parametrize("hour,expected_night", [
        (3, 1), (22, 1), (23, 1), (0, 1), (14, 0), (10, 0), (6, 0),
    ])
    def test_is_night_flag(self, hour, expected_night):
        df = pd.DataFrame([{
            "transaction_amount": 50, "merchant_category": "grocery",
            "hour_of_day": hour, "day_of_week": 2, "distance_from_home": 10,
            "is_online": 0, "card_age_days": 365,
            "num_transactions_last_hour": 1, "amount_vs_avg_ratio": 1.0,
            "is_fraud": 0,
        }])
        result, _ = fraud_features(df)
        assert result["is_night"].iloc[0] == expected_night

    @pytest.mark.parametrize("day,expected_weekend", [
        (0, 0), (4, 0), (5, 1), (6, 1),
    ])
    def test_is_weekend_flag(self, day, expected_weekend):
        df = pd.DataFrame([{
            "transaction_amount": 50, "merchant_category": "grocery",
            "hour_of_day": 14, "day_of_week": day, "distance_from_home": 10,
            "is_online": 0, "card_age_days": 365,
            "num_transactions_last_hour": 1, "amount_vs_avg_ratio": 1.0,
            "is_fraud": 0,
        }])
        result, _ = fraud_features(df)
        assert result["is_weekend"].iloc[0] == expected_weekend

    def test_no_nan_in_features(self, fraud_df):
        result, _ = fraud_features(fraud_df)
        feature_data = result[fraud_cols()]
        assert feature_data.notna().all().all()


# ---------------------------------------------------------------------------
# Housing Features
# ---------------------------------------------------------------------------


class TestHousingFeatures:

    def test_all_feature_columns_present(self, housing_df):
        result = housing_features(housing_df)
        for col in housing_cols():
            assert col in result.columns, f"Missing: {col}"

    def test_property_age_value(self):
        df = pd.DataFrame([{
            "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
            "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["property_age"].iloc[0] == 24

    def test_total_rooms(self):
        df = pd.DataFrame([{
            "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
            "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["total_rooms"].iloc[0] == 5

    def test_sqft_per_room(self):
        df = pd.DataFrame([{
            "square_feet": 2000, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
            "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["sqft_per_room"].iloc[0] == pytest.approx(400.0)

    def test_lot_to_house_ratio(self):
        df = pd.DataFrame([{
            "square_feet": 2000, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 10000, "garage_spaces": 2,
            "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["lot_to_house_ratio"].iloc[0] == pytest.approx(5.0)

    @pytest.mark.parametrize("garage,expected", [(0, 0), (1, 1), (3, 1)])
    def test_has_garage_flag(self, garage, expected):
        df = pd.DataFrame([{
            "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": garage,
            "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["has_garage"].iloc[0] == expected

    @pytest.mark.parametrize("pool,tier,expected", [
        (1, 4, 1), (1, 5, 1), (0, 5, 0), (1, 3, 0), (0, 3, 0),
    ])
    def test_is_luxury_flag(self, pool, tier, expected):
        df = pd.DataFrame([{
            "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
            "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
            "has_pool": pool, "neighborhood_tier": tier, "proximity_to_city_center": 10,
        }])
        result = housing_features(df)
        assert result["is_luxury"].iloc[0] == expected

    def test_no_nan_in_features(self, housing_df):
        result = housing_features(housing_df)
        feature_data = result[housing_cols()]
        assert feature_data.notna().all().all()


# ---------------------------------------------------------------------------
# Timeseries Features
# ---------------------------------------------------------------------------


class TestTimeseriesFeatures:

    def test_create_sequences_shapes(self):
        data = np.arange(100, dtype=float)
        X, y = create_sequences(data, window_size=30, forecast_horizon=7)
        assert X.ndim == 3
        assert X.shape[1] == 30
        assert X.shape[2] == 1
        assert y.ndim == 2
        assert y.shape[1] == 7
        assert X.shape[0] == y.shape[0]

    def test_create_sequences_count(self):
        data = np.arange(50, dtype=float)
        X, y = create_sequences(data, window_size=10, forecast_horizon=5)
        expected = 50 - 10 - 5 + 1
        assert X.shape[0] == expected

    def test_create_sequences_values(self):
        data = np.arange(20, dtype=float)
        X, y = create_sequences(data, window_size=5, forecast_horizon=3)
        np.testing.assert_array_equal(X[0, :, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(y[0], [5, 6, 7])

    def test_demand_dataset_length(self):
        X = np.random.randn(10, 30, 1).astype(np.float32)
        y = np.random.randn(10, 7).astype(np.float32)
        ds = DemandDataset(X, y)
        assert len(ds) == 10

    def test_demand_dataset_item_types(self):
        import torch
        X = np.random.randn(5, 30, 1).astype(np.float32)
        y = np.random.randn(5, 7).astype(np.float32)
        ds = DemandDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (30, 1)
        assert y_item.shape == (7,)
