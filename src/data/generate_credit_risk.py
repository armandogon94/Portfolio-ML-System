"""Generate synthetic credit risk dataset with realistic distributions."""

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)


def generate_credit_risk_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan application data.

    Features are generated with realistic correlations:
    - Higher income correlates with higher credit scores
    - Higher debt-to-income correlates with higher default rates
    - Payment history strongly predicts default
    """
    rng = np.random.default_rng(seed)

    # Age: normal distribution centered at 40
    age = rng.normal(40, 12, n_samples).clip(18, 75).astype(int)

    # Annual income: log-normal (right-skewed, realistic)
    annual_income = rng.lognormal(mean=10.8, sigma=0.6, size=n_samples).clip(15000, 500000)

    # Credit score: correlated with income (higher income -> slightly higher score)
    income_factor = (annual_income - annual_income.mean()) / annual_income.std() * 30
    credit_score = (rng.normal(680, 80, n_samples) + income_factor).clip(300, 850).astype(int)

    # Number of open accounts: Poisson distribution
    num_open_accounts = rng.poisson(4, n_samples).clip(0, 20)

    # Payment history percentage: beta distribution (skewed toward high values)
    payment_history_pct = rng.beta(8, 2, n_samples) * 100

    # Debt-to-income ratio: gamma distribution (right-skewed)
    debt_to_income_ratio = rng.gamma(2, 0.15, n_samples).clip(0, 1.5)

    # Employment years: correlated with age
    max_emp_years = (age - 18).clip(0)
    employment_years = (rng.exponential(8, n_samples)).clip(0, max_emp_years)

    # Loan amount requested: log-normal
    loan_amount = rng.lognormal(mean=9.5, sigma=0.8, size=n_samples).clip(1000, 200000)

    # Generate default target using logistic function of features
    # Higher credit score, income, payment history -> lower default probability
    # Higher debt-to-income, loan amount -> higher default probability
    z = (
        -3.0
        + 0.02 * (40 - age)
        - 0.8 * ((annual_income - 60000) / 60000)
        - 1.5 * ((credit_score - 680) / 100)
        + 0.3 * ((num_open_accounts - 4) / 3)
        - 1.2 * ((payment_history_pct - 75) / 25)
        + 2.0 * ((debt_to_income_ratio - 0.3) / 0.3)
        - 0.2 * ((employment_years - 8) / 8)
        + 0.5 * ((loan_amount - 15000) / 15000)
        + rng.normal(0, 0.5, n_samples)  # noise
    )
    default_prob = 1 / (1 + np.exp(-z))
    is_default = (rng.random(n_samples) < default_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": np.round(annual_income, 2),
        "credit_score": credit_score,
        "num_open_accounts": num_open_accounts,
        "payment_history_pct": np.round(payment_history_pct, 1),
        "debt_to_income_ratio": np.round(debt_to_income_ratio, 3),
        "employment_years": np.round(employment_years, 1),
        "loan_amount": np.round(loan_amount, 2),
        "is_default": is_default,
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_credit_risk_data()
    output_path = get_project_root() / "data" / "raw" / "credit_risk.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} credit risk samples -> {output_path}")
    print(f"Default rate: {df['is_default'].mean():.1%}")
