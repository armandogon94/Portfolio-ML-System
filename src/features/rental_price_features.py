"""Feature engineering for rental-price regression (Real Estate / A.3).

Kept deliberately simple: the synthetic generator already models the
relationship between raw features and nightly rate, so no derived features
are needed to hit a reasonable R^2. This module exists so the trainer and
predictor can call the same ``engineer_features`` / ``get_feature_columns``
pair every other model uses — keeping the pipeline shape uniform across
industries.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged — no derived features for rental price."""
    return df.copy()


def get_feature_columns() -> list[str]:
    """Return the 8 feature columns used by the rental-price model."""
    return [
        "bedrooms",
        "bathrooms",
        "square_feet",
        "property_type",
        "location_tier",
        "distance_to_downtown_km",
        "amenity_score",
        "peer_nightly_rate",
    ]
