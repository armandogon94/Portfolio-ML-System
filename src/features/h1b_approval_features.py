"""Feature engineering for the H-1B approval classifier.

Engineered columns deliberately stay thin — the 8 raw features already carry
most of the signal, so we only add ratios/interactions that encode domain
intuition (wage per SOC level, senior-specialty bonus).
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for the H-1B approval model."""
    df = df.copy()

    # Wage relative to SOC tier — a $100k role at SOC level 1 is suspicious,
    # at SOC level 4 is normal. The divisor keeps the feature O(1).
    df["wage_per_soc_level"] = df["prevailing_wage"] / (df["soc_code_level"] * 30_000).clip(lower=1)

    # Education × SOC code interaction — PhDs working at low SOC levels are
    # anomalous and should be down-weighted by the model.
    df["education_soc_product"] = df["education_level"] * df["soc_code_level"]

    # Seniority composite — job level + experience bucket.
    df["experience_bucket"] = pd.cut(
        df["experience_years"],
        bins=[-1, 2, 5, 10, 20, 40],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["seniority_score"] = df["job_level"] + df["experience_bucket"]

    return df


def get_feature_columns() -> list[str]:
    """Return all feature columns consumed by the model (raw + engineered)."""
    return [
        "prevailing_wage",
        "soc_code_level",
        "employer_size_tier",
        "job_level",
        "education_level",
        "experience_years",
        "country_of_citizenship_tier",
        "employer_prior_approval_rate",
        "wage_per_soc_level",
        "education_soc_product",
        "experience_bucket",
        "seniority_score",
    ]
