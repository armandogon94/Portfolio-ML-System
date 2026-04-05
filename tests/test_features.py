"""Tests for feature engineering."""

import pandas as pd

from src.features.credit_risk_features import engineer_features as credit_features
from src.features.credit_risk_features import get_feature_columns as credit_cols
from src.features.housing_features import engineer_features as housing_features
from src.features.housing_features import get_feature_columns as housing_cols


def test_credit_risk_features():
    df = pd.DataFrame([{
        "age": 35, "annual_income": 65000, "credit_score": 700,
        "num_open_accounts": 3, "payment_history_pct": 85,
        "debt_to_income_ratio": 0.3, "employment_years": 8, "loan_amount": 25000,
    }])
    result = credit_features(df)
    assert "loan_to_income" in result.columns
    assert "credit_utilization" in result.columns
    assert "employment_stability" in result.columns
    assert "credit_tier" in result.columns
    # All feature columns should exist
    for col in credit_cols():
        assert col in result.columns


def test_housing_features():
    df = pd.DataFrame([{
        "square_feet": 1800, "bedrooms": 3, "bathrooms": 2,
        "year_built": 2000, "lot_size_sqft": 8000, "garage_spaces": 2,
        "has_pool": 0, "neighborhood_tier": 3, "proximity_to_city_center": 10,
    }])
    result = housing_features(df)
    assert "property_age" in result.columns
    assert "sqft_per_room" in result.columns
    assert "is_luxury" in result.columns
    for col in housing_cols():
        assert col in result.columns
