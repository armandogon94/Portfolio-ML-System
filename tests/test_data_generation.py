"""Tests for synthetic data generation."""

import pandas as pd

from src.data.generate_credit_risk import generate_credit_risk_data
from src.data.generate_fraud import generate_fraud_data
from src.data.generate_housing import generate_housing_data
from src.data.generate_timeseries import generate_timeseries_data


def test_credit_risk_data():
    df = generate_credit_risk_data(n_samples=1000, seed=42)
    assert len(df) == 1000
    assert "is_default" in df.columns
    assert df["age"].between(18, 75).all()
    assert df["credit_score"].between(300, 850).all()
    assert df["is_default"].isin([0, 1]).all()
    # Default rate should be roughly 5-15%
    rate = df["is_default"].mean()
    assert 0.01 < rate < 0.30


def test_fraud_data():
    df = generate_fraud_data(n_samples=10000, seed=42)
    assert len(df) == 10000
    assert "is_fraud" in df.columns
    assert df["is_fraud"].isin([0, 1]).all()
    # Fraud rate should be ~2%
    rate = df["is_fraud"].mean()
    assert 0.01 < rate < 0.05


def test_housing_data():
    df = generate_housing_data(n_samples=1000, seed=42)
    assert len(df) == 1000
    assert "price" in df.columns
    assert df["price"].gt(0).all()
    assert df["bedrooms"].between(1, 6).all()
    assert df["neighborhood_tier"].between(1, 5).all()


def test_timeseries_data():
    df = generate_timeseries_data(n_years=1, seed=42)
    assert "date" in df.columns
    assert "demand" in df.columns
    assert "product_category" in df.columns
    assert df["product_category"].nunique() == 5
    assert df["demand"].ge(0).all()
