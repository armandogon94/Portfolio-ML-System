"""Feature engineering for fraud detection."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepare features for fraud detection model.

    Returns processed dataframe and artifacts dict (encoders, scaler).
    """
    df = df.copy()
    artifacts = {}

    # Encode merchant category
    le = LabelEncoder()
    df["merchant_category_encoded"] = le.fit_transform(df["merchant_category"])
    artifacts["merchant_encoder"] = le

    # Log-transform amount (reduces skew)
    df["log_amount"] = df["transaction_amount"].clip(lower=0.01).apply(lambda x: __import__("numpy").log1p(x))

    # Time features
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df, artifacts


def get_feature_columns() -> list[str]:
    """Return numerical feature columns for the autoencoder."""
    return [
        "transaction_amount", "log_amount", "merchant_category_encoded",
        "hour_of_day", "day_of_week", "distance_from_home", "is_online",
        "card_age_days", "num_transactions_last_hour", "amount_vs_avg_ratio",
        "is_night", "is_weekend",
    ]
