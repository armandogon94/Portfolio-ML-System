"""Feature engineering for credit risk scoring."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for credit risk model."""
    df = df.copy()

    # Loan-to-income ratio
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)

    # Credit utilization proxy (accounts * debt_to_income)
    df["credit_utilization"] = df["num_open_accounts"] * df["debt_to_income_ratio"]

    # Income stability proxy (employment_years / age fraction of working life)
    working_years = (df["age"] - 18).clip(lower=1)
    df["employment_stability"] = df["employment_years"] / working_years

    # Risk bucket based on credit score
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 800, 900],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Return all feature columns used by the model."""
    return [
        "age", "annual_income", "credit_score", "num_open_accounts",
        "payment_history_pct", "debt_to_income_ratio", "employment_years",
        "loan_amount", "loan_to_income", "credit_utilization",
        "employment_stability", "credit_tier",
    ]
