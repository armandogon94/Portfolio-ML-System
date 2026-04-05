"""Feature engineering for housing price prediction."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for housing price model."""
    df = df.copy()

    # Age of property
    df["property_age"] = 2024 - df["year_built"]

    # Price per square foot proxy features
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["sqft_per_room"] = df["square_feet"] / df["total_rooms"].clip(lower=1)

    # Lot-to-house ratio
    df["lot_to_house_ratio"] = df["lot_size_sqft"] / df["square_feet"].clip(lower=1)

    # Has garage flag
    df["has_garage"] = (df["garage_spaces"] > 0).astype(int)

    # Luxury indicator (pool + upscale neighborhood)
    df["is_luxury"] = ((df["has_pool"] == 1) & (df["neighborhood_tier"] >= 4)).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Return all feature columns used by the model."""
    return [
        "square_feet", "bedrooms", "bathrooms", "year_built", "lot_size_sqft",
        "garage_spaces", "has_pool", "neighborhood_tier", "proximity_to_city_center",
        "property_age", "total_rooms", "sqft_per_room", "lot_to_house_ratio",
        "has_garage", "is_luxury",
    ]
