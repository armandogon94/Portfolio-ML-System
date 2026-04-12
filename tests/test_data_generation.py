"""Tests for synthetic data generation — parametrized across sample sizes."""

import numpy as np
import pandas as pd
import pytest

from src.data.generate_credit_risk import generate_credit_risk_data
from src.data.generate_fraud import MERCHANT_CATEGORIES, generate_fraud_data
from src.data.generate_housing import generate_housing_data
from src.data.generate_timeseries import PRODUCT_CATEGORIES, generate_timeseries_data

# ---------------------------------------------------------------------------
# Credit Risk Data
# ---------------------------------------------------------------------------

CREDIT_RISK_COLUMNS = [
    "age", "annual_income", "credit_score", "num_open_accounts",
    "payment_history_pct", "debt_to_income_ratio", "employment_years",
    "loan_amount", "is_default",
]


@pytest.mark.parametrize("n_samples", [100, 1000, 5000])
class TestCreditRiskData:

    def test_row_count(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert len(df) == n_samples

    def test_column_completeness(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        for col in CREDIT_RISK_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert set(df.columns) == set(CREDIT_RISK_COLUMNS)

    def test_age_range(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df["age"].between(18, 75).all()

    def test_credit_score_range(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df["credit_score"].between(300, 850).all()

    def test_income_positive(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df["annual_income"].gt(0).all()

    def test_target_binary(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df["is_default"].isin([0, 1]).all()

    def test_default_rate_range(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        rate = df["is_default"].mean()
        assert 0.01 < rate < 0.30, f"Default rate {rate:.2%} outside expected range"

    def test_no_nulls(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df.notna().all().all()

    def test_dtypes(self, n_samples):
        df = generate_credit_risk_data(n_samples=n_samples, seed=42)
        assert df["age"].dtype in (np.int64, np.int32, int)
        assert df["credit_score"].dtype in (np.int64, np.int32, int)
        assert np.issubdtype(df["annual_income"].dtype, np.floating)


def test_credit_risk_reproducibility():
    df1 = generate_credit_risk_data(n_samples=100, seed=42)
    df2 = generate_credit_risk_data(n_samples=100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Fraud Data
# ---------------------------------------------------------------------------

FRAUD_COLUMNS = [
    "transaction_amount", "merchant_category", "hour_of_day", "day_of_week",
    "distance_from_home", "is_online", "card_age_days",
    "num_transactions_last_hour", "amount_vs_avg_ratio", "is_fraud",
]


@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
class TestFraudData:

    def test_row_count(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert len(df) == n_samples

    def test_column_completeness(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        for col in FRAUD_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_binary(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df["is_fraud"].isin([0, 1]).all()

    def test_fraud_rate(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        rate = df["is_fraud"].mean()
        assert 0.01 < rate < 0.05, f"Fraud rate {rate:.2%} outside expected range"

    def test_amounts_positive(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df["transaction_amount"].gt(0).all()

    def test_hour_range(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df["hour_of_day"].between(0, 23).all()

    def test_day_of_week_range(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df["day_of_week"].between(0, 6).all()

    def test_merchant_categories_valid(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df["merchant_category"].isin(MERCHANT_CATEGORIES).all()

    def test_no_nulls(self, n_samples):
        df = generate_fraud_data(n_samples=n_samples, seed=42)
        assert df.notna().all().all()


# ---------------------------------------------------------------------------
# Housing Data
# ---------------------------------------------------------------------------

HOUSING_COLUMNS = [
    "square_feet", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
    "garage_spaces", "has_pool", "neighborhood_tier", "proximity_to_city_center", "price",
]


@pytest.mark.parametrize("n_samples", [100, 1000, 5000])
class TestHousingData:

    def test_row_count(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert len(df) == n_samples

    def test_column_completeness(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        for col in HOUSING_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_price_positive(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["price"].gt(0).all()

    def test_bedroom_range(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["bedrooms"].between(1, 6).all()

    def test_neighborhood_tier_range(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["neighborhood_tier"].between(1, 5).all()

    def test_has_pool_binary(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["has_pool"].isin([0, 1]).all()

    def test_year_built_range(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["year_built"].between(1950, 2024).all()

    def test_no_nulls(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df.notna().all().all()

    def test_dtypes(self, n_samples):
        df = generate_housing_data(n_samples=n_samples, seed=42)
        assert df["square_feet"].dtype in (np.int64, np.int32, int)
        assert df["price"].dtype in (np.int64, np.int32, int)


# ---------------------------------------------------------------------------
# Timeseries Data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_years", [1, 2])
class TestTimeseriesData:

    def test_has_required_columns(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        for col in ["date", "product_category", "demand", "is_holiday"]:
            assert col in df.columns

    def test_product_count(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        assert df["product_category"].nunique() == len(PRODUCT_CATEGORIES)

    def test_demand_non_negative(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        assert df["demand"].ge(0).all()

    def test_row_count(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        expected = n_years * 365 * len(PRODUCT_CATEGORIES)
        assert len(df) == expected

    def test_date_range(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        n_days = n_years * 365
        assert df["date"].nunique() == n_days

    def test_no_nulls(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        assert df.notna().all().all()

    def test_holiday_flag_binary(self, n_years):
        df = generate_timeseries_data(n_years=n_years, seed=42)
        assert df["is_holiday"].isin([0, 1]).all()
