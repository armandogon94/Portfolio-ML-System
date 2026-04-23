"""Feature engineering for delivery ETA regression.

Pass-through pipeline — the raw generator already emits clean numeric features
that XGBoost handles natively, so there's no transformation layer (unlike the
credit-risk / housing features). Kept as a module for symmetry with the rest
of the Phase A codebase and to give future versions a dedicated place to add
derived features (e.g., distance × congestion, weekend flag) without touching
the trainer.
"""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` unchanged.

    The delivery-ETA generator already emits model-ready numeric columns, so
    this function is a pass-through. Returning a ``.copy()`` keeps downstream
    callers from mutating the caller's frame.
    """
    return df.copy()


def get_feature_columns() -> list[str]:
    """Ordered list of features fed to the XGBoost regressor."""
    return [
        "distance_km",
        "package_weight_kg",
        "traffic_congestion",
        "weather_severity",
        "time_of_day",
        "day_of_week",
        "carrier_priority",
        "origin_destination_tier",
    ]
