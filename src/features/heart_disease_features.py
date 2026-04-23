"""Feature engineering for heart disease risk classification.

Pass-through: the 8 clinical inputs are already numeric and on scales
LightGBM handles well, so no derived features are added. The module
still exists so the trainer follows the same shape as other industries
and future enhancements (e.g. age-adjusted thalach buckets) have a
natural home.
"""

from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` unchanged — heart-disease features are pass-through.

    Kept as a function (not a no-op alias) so tests can monkeypatch it
    and so adding derived columns later is a localised edit.
    """
    return df.copy()


def get_feature_columns() -> list[str]:
    """Return all feature columns used by the model (input order matters)."""
    return [
        "age",
        "sex",
        "chest_pain_type",
        "resting_bp",
        "cholesterol",
        "max_heart_rate",
        "exercise_angina",
        "oldpeak",
    ]
