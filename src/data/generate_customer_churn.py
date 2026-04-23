"""Generate synthetic bank-customer churn dataset.

Targets ~20% churn with a logistic model over 8 features. Key signals:
- tenure_months: strong negative (loyal customers don't leave)
- is_active_member: strong negative (engaged customers stay)
- age: weak positive (older cohorts churn slightly more)
- num_products: U-shaped — 1 product (under-invested) or 4+ (over-sold)
  churn more; 2–3 is the retention sweet spot
- balance: extremes drive churn (dormant zero balances and high-wealth
  flight risks) more than mid-tier customers
"""

import numpy as np
import pandas as pd


def generate_customer_churn_data(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bank-customer churn data.

    The generator composes a logistic score from eight features and samples
    the binary target from that probability. Parameters are tuned so the
    resulting churn rate lands around 20% with moderate class separability
    suitable for an XGBoost demo.
    """
    rng = np.random.default_rng(seed)

    # Tenure: how long the customer has banked with us, in months.
    # Uniform so we can observe the full tenure gradient in the demo.
    tenure_months = rng.integers(0, 121, n_samples)

    # Balance: right-skewed (many near-zero), capped at 250k.
    # Using log-normal gives a realistic "dormant accounts cluster near 0"
    # distribution while still producing some wealthy outliers.
    balance = rng.lognormal(mean=10.0, sigma=1.3, size=n_samples).clip(0, 250_000)
    # Inject a dormant-account cluster: ~15% of customers with zero balance.
    balance = np.where(rng.random(n_samples) < 0.15, 0.0, balance)

    # Number of products held (credit card, savings, mortgage, etc.).
    # Poisson clipped to [1, 6] — most have 1–2, some up to 6.
    num_products = (rng.poisson(1.6, n_samples) + 1).clip(1, 6)

    # Binary flags.
    has_credit_card = (rng.random(n_samples) < 0.70).astype(int)
    is_active_member = (rng.random(n_samples) < 0.52).astype(int)

    # Estimated salary: broadly uniform within plausible band.
    estimated_salary = rng.uniform(10_000, 200_000, n_samples)

    # Age: normal centered at 40, clipped to [18, 92].
    age = rng.normal(40, 11, n_samples).clip(18, 92).astype(int)

    # Geography tier: 1 = core market, 2 = established, 3 = expanding.
    # Expanding markets churn more (higher competition for new accounts).
    geography_tier = rng.choice([1, 2, 3], size=n_samples, p=[0.45, 0.35, 0.20])

    # U-shaped product penalty: 1 or 4+ → positive; 2–3 → negative.
    # We encode it as a piecewise scalar added to the logistic score.
    product_signal = np.where(
        num_products == 1, 0.6,
        np.where(num_products <= 3, -0.5, 0.9),
    )

    # Balance signal: high-wealth (>150k) or zero balance → positive churn
    # push; mid-tier → slight negative. Normalize around 75k so the mid
    # range is neutral-to-retentive.
    balance_signal = np.where(
        balance == 0, 0.6,
        np.where(balance > 150_000, 0.5, -0.2 * ((balance - 75_000) / 75_000)),
    )

    # Logistic score: negative = retain, positive = churn.
    # Intercept tuned to target ~20% overall churn rate.
    z = (
        -0.7
        - 0.012 * tenure_months                  # loyalty compounds
        - 1.3 * is_active_member                 # activity is the strongest retainer
        + 0.012 * (age - 40)                     # weak age lift
        + product_signal
        + balance_signal
        + 0.25 * (geography_tier - 1)            # expanding markets leak customers
        - 0.1 * has_credit_card                  # CC holders slightly stickier
        + 0.000002 * (estimated_salary - 75_000) # noise-level salary effect
        + rng.normal(0, 0.45, n_samples)         # irreducible noise
    )
    churn_prob = 1 / (1 + np.exp(-z))
    churned = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months.astype(int),
        "balance": np.round(balance, 2),
        "num_products": num_products.astype(int),
        "has_credit_card": has_credit_card,
        "is_active_member": is_active_member,
        "estimated_salary": np.round(estimated_salary, 2),
        "age": age,
        "geography_tier": geography_tier.astype(int),
        "churned": churned,
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_customer_churn_data()
    output_path = get_project_root() / "data" / "raw" / "customer_churn.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customer churn samples -> {output_path}")
    print(f"Churn rate: {df['churned'].mean():.1%}")
