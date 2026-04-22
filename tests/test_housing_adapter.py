"""Tests for the Zillow -> canonical housing schema adapter.

Verifies that the adapter produces the exact canonical schema consumed by
`src/features/housing_features.py`, regardless of which Zillow column names
are present in the input.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.adapters import housing_adapter
from src.data.generate_housing import generate_housing_data

CANONICAL_COLUMNS = [
    "square_feet",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size_sqft",
    "garage_spaces",
    "has_pool",
    "neighborhood_tier",
    "proximity_to_city_center",
    "price",
]


class TestHousingAdapterRenames:

    def test_adapter_renames_zillow_columns(self):
        """Typical Zillow column names are mapped to the canonical schema."""
        zillow_df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 2,
                "HasPool": 0,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
            }
        ])

        result = housing_adapter(zillow_df)

        for col in CANONICAL_COLUMNS:
            assert col in result.columns, f"Missing canonical column: {col}"
        assert result["square_feet"].iloc[0] == 1800
        assert result["bedrooms"].iloc[0] == 3
        assert result["price"].iloc[0] == 350000
        assert result["proximity_to_city_center"].iloc[0] == 10.0

    def test_adapter_handles_alternative_column_names(self):
        """Lowercase / snake_case Zillow alternatives are also mapped."""
        alt_df = pd.DataFrame([
            {
                "sqft_living": 2200,
                "beds": 4,
                "baths": 3,
                "yr_built": 1995,
                "sqft_lot": 9500,
                "garage_cars": 2,
                "pool": 1,
                "neighborhood_tier": 4,
                "distance_to_center": 8.5,
                "Price": 500000,
            }
        ])

        result = housing_adapter(alt_df)

        assert list(result.columns) == CANONICAL_COLUMNS
        assert result["square_feet"].iloc[0] == 2200
        assert result["has_pool"].iloc[0] == 1
        assert result["neighborhood_tier"].iloc[0] == 4


class TestHousingAdapterImputesMissing:

    def test_has_pool_defaults_to_zero_when_missing(self):
        df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 1,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
            }
        ])

        result = housing_adapter(df)

        assert "has_pool" in result.columns
        assert (result["has_pool"] == 0).all()

    def test_neighborhood_tier_defaults_to_three_when_missing(self):
        df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 1,
                "HasPool": 0,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
            }
        ])

        result = housing_adapter(df)

        assert "neighborhood_tier" in result.columns
        assert (result["neighborhood_tier"] == 3).all()

    def test_missing_garage_spaces_defaults_to_zero(self):
        """When no garage source column is present at all, fill with 0."""
        df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "HasPool": 0,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
            }
        ])

        result = housing_adapter(df)

        assert "garage_spaces" in result.columns
        assert (result["garage_spaces"] == 0).all()

    def test_no_key_error_for_fully_missing_numeric_column(self):
        """Even if a source numeric column is completely absent, no KeyError."""
        df = pd.DataFrame([
            {
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "SalePrice": 350000,
            }
        ])

        # Should not raise — missing columns get imputed
        result = housing_adapter(df)

        assert list(result.columns) == CANONICAL_COLUMNS


class TestHousingAdapterSchema:

    def test_adapter_drops_extra_columns(self):
        """Unrelated Zillow columns (Zipcode, FloodZone, ListingAgent) are dropped."""
        df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 2,
                "HasPool": 0,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
                # Unrelated Zillow columns
                "Zipcode": "94107",
                "FloodZone": "X",
                "ListingAgent": "Alice Example",
                "MLSNumber": "ML12345",
            }
        ])

        result = housing_adapter(df)

        assert set(result.columns) == set(CANONICAL_COLUMNS)
        assert "Zipcode" not in result.columns
        assert "FloodZone" not in result.columns
        assert "ListingAgent" not in result.columns

    def test_adapter_output_schema_matches_synthetic(self):
        """Adapter output columns equal generate_housing_data() columns in exact order."""
        synthetic = generate_housing_data(n_samples=10, seed=0)

        zillow_df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 2,
                "HasPool": 0,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
            }
        ])
        result = housing_adapter(zillow_df)

        assert list(result.columns) == list(synthetic.columns)
        assert list(result.columns) == CANONICAL_COLUMNS


class TestHousingAdapterEdgeCases:

    def test_empty_dataframe_returns_empty_with_canonical_columns(self):
        """An empty DataFrame yields an empty DataFrame with canonical columns."""
        result = housing_adapter(pd.DataFrame())

        assert list(result.columns) == CANONICAL_COLUMNS
        assert len(result) == 0

    def test_adapter_preserves_row_count(self):
        """Adapter must not drop rows."""
        df = pd.DataFrame({
            "SquareFootage": [1800, 2200, 1500, 3000, 1100],
            "Bedrooms": [3, 4, 2, 5, 2],
            "Bathrooms": [2, 3, 1, 4, 1],
            "YearBuilt": [2000, 1995, 1980, 2010, 1960],
            "LotSize": [8000, 9500, 5000, 12000, 4000],
            "GarageSpaces": [2, 2, 1, 3, 0],
            "HasPool": [0, 1, 0, 1, 0],
            "NeighborhoodTier": [3, 4, 2, 5, 1],
            "DistanceToCBD": [10.0, 8.5, 15.0, 5.0, 25.0],
            "SalePrice": [350000, 500000, 220000, 850000, 150000],
        })

        result = housing_adapter(df)

        assert len(result) == 5
        assert list(result.columns) == CANONICAL_COLUMNS

    def test_adapter_does_not_mutate_input(self):
        """Adapter should not mutate the caller's DataFrame."""
        df = pd.DataFrame([
            {
                "SquareFootage": 1800,
                "Bedrooms": 3,
                "Bathrooms": 2,
                "YearBuilt": 2000,
                "LotSize": 8000,
                "GarageSpaces": 2,
                "HasPool": 0,
                "NeighborhoodTier": 3,
                "DistanceToCBD": 10.0,
                "SalePrice": 350000,
                "Zipcode": "94107",
            }
        ])
        before = df.copy()

        housing_adapter(df)

        pd.testing.assert_frame_equal(df, before)

    def test_numeric_missing_column_imputed_with_median_when_partial(self):
        """If a canonical column is present but has NaNs, fill with median."""
        # Adapter operates at the column-presence level: if a column is absent
        # entirely, we impute with 0 / default. But if the source column exists
        # with NaN values, we fill them with the median of the non-null values.
        df = pd.DataFrame({
            "SquareFootage": [1800, 2000, None, 2200, 1500],
            "Bedrooms": [3, 4, 2, 5, 2],
            "Bathrooms": [2, 3, 1, 4, 1],
            "YearBuilt": [2000, 1995, 1980, 2010, 1960],
            "LotSize": [8000, 9500, 5000, 12000, 4000],
            "GarageSpaces": [2, 2, 1, 3, 0],
            "HasPool": [0, 1, 0, 1, 0],
            "NeighborhoodTier": [3, 4, 2, 5, 1],
            "DistanceToCBD": [10.0, 8.5, 15.0, 5.0, 25.0],
            "SalePrice": [350000, 500000, 220000, 850000, 150000],
        })

        result = housing_adapter(df)

        # The NaN square_feet value should be filled with the median (1900)
        assert result["square_feet"].notna().all()
        # Median of [1800, 2000, 2200, 1500] = 1900
        assert result["square_feet"].iloc[2] == pytest.approx(1900)

    def test_partial_nan_in_categorical_uses_default(self):
        """If has_pool has NaNs, fill with the categorical default (0)."""
        df = pd.DataFrame({
            "SquareFootage": [1800, 2000, 2200],
            "Bedrooms": [3, 4, 2],
            "Bathrooms": [2, 3, 1],
            "YearBuilt": [2000, 1995, 1980],
            "LotSize": [8000, 9500, 5000],
            "GarageSpaces": [2, 2, 1],
            "HasPool": [1, None, 0],
            "NeighborhoodTier": [3, 4, 2],
            "DistanceToCBD": [10.0, 8.5, 15.0],
            "SalePrice": [350000, 500000, 220000],
        })

        result = housing_adapter(df)

        assert result["has_pool"].notna().all()
        # Middle row's NaN should be filled with the default (0)
        assert result["has_pool"].iloc[1] == 0

    def test_all_nan_numeric_column_fills_with_zero(self):
        """If a numeric column exists but is entirely NaN, fall back to 0."""
        df = pd.DataFrame({
            "SquareFootage": [None, None, None],
            "Bedrooms": [3, 4, 2],
            "Bathrooms": [2, 3, 1],
            "YearBuilt": [2000, 1995, 1980],
            "LotSize": [8000, 9500, 5000],
            "GarageSpaces": [2, 2, 1],
            "HasPool": [0, 1, 0],
            "NeighborhoodTier": [3, 4, 2],
            "DistanceToCBD": [10.0, 8.5, 15.0],
            "SalePrice": [350000, 500000, 220000],
        })

        result = housing_adapter(df)

        assert result["square_feet"].notna().all()
        assert (result["square_feet"] == 0).all()
