"""Feature engineering for dental patient no-show prediction.

Pass-through for Phase A.4 — the generator already emits clean numeric
features. Stub left here so trainer + predictor can import a consistent
API (``engineer_features`` + ``get_feature_columns``) across industries.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy. No derived features for A.4."""
    return df.copy()


def get_feature_columns() -> list[str]:
    """Columns consumed by the XGBoost classifier (order matters)."""
    return [
        "age",
        "prior_no_shows",
        "days_until_appointment",
        "appointment_hour",
        "distance_km",
        "insurance_type",
        "procedure_complexity",
        "prior_appointments",
    ]
