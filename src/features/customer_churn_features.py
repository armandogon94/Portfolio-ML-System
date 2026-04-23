"""Feature engineering for customer-churn prediction.

The 8 raw inputs are already well-scaled and semantically distinct, so
this pipeline is a pass-through. Kept as a dedicated module to match the
project convention (one feature module per problem) and to give a stable
import target if richer engineering is added later.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the DataFrame unchanged (defensive copy)."""
    return df.copy()


def get_feature_columns() -> list[str]:
    """Return the ordered feature columns the model consumes."""
    return [
        "tenure_months",
        "balance",
        "num_products",
        "has_credit_card",
        "is_active_member",
        "estimated_salary",
        "age",
        "geography_tier",
    ]
