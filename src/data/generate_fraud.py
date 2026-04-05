"""Generate synthetic fraud transaction dataset with realistic patterns."""

import numpy as np
import pandas as pd


MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "clothing", "travel", "entertainment", "healthcare", "utilities",
    "education", "home_improvement", "automotive", "subscription", "atm_withdrawal",
]


def generate_fraud_data(n_samples: int = 200000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic transaction data with ~2% fraud rate.

    Fraudulent transactions have shifted distributions:
    - Higher amounts, unusual hours, greater distances
    - More online transactions, unusual merchant categories
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * 0.02)
    n_normal = n_samples - n_fraud

    # --- Normal transactions ---
    normal = _generate_transactions(rng, n_normal, is_fraud=False)

    # --- Fraudulent transactions ---
    fraud = _generate_transactions(rng, n_fraud, is_fraud=True)

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def _generate_transactions(rng: np.random.Generator, n: int, is_fraud: bool) -> pd.DataFrame:
    """Generate transaction records with fraud-specific distributions."""

    if is_fraud:
        # Fraud: higher amounts, unusual hours, farther distances
        transaction_amount = rng.lognormal(mean=5.5, sigma=1.5, size=n).clip(0.01, 10000)
        hour_of_day = rng.choice(24, n, p=_fraud_hour_distribution())
        distance_from_home = rng.exponential(80, n).clip(0, 500)
        is_online = rng.binomial(1, 0.7, n)  # 70% online
        merchant_idx = rng.choice(len(MERCHANT_CATEGORIES), n, p=_fraud_merchant_distribution())
        num_transactions_last_hour = rng.poisson(3, n).clip(0, 20)
    else:
        # Normal: typical patterns
        transaction_amount = rng.lognormal(mean=3.5, sigma=1.0, size=n).clip(0.01, 5000)
        hour_of_day = rng.choice(24, n, p=_normal_hour_distribution())
        distance_from_home = rng.exponential(10, n).clip(0, 200)
        is_online = rng.binomial(1, 0.3, n)  # 30% online
        merchant_idx = rng.choice(len(MERCHANT_CATEGORIES), n, p=_normal_merchant_distribution())
        num_transactions_last_hour = rng.poisson(1, n).clip(0, 10)

    merchant_category = [MERCHANT_CATEGORIES[i] for i in merchant_idx]
    day_of_week = rng.integers(0, 7, n)
    card_age_days = rng.integers(30, 3650, n)

    # Amount vs average ratio (fraud tends to be higher)
    avg_amount = 50.0 if not is_fraud else 50.0  # Relative to "normal" average
    amount_vs_avg_ratio = transaction_amount / avg_amount

    return pd.DataFrame({
        "transaction_amount": np.round(transaction_amount, 2),
        "merchant_category": merchant_category,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "distance_from_home": np.round(distance_from_home, 1),
        "is_online": is_online,
        "card_age_days": card_age_days,
        "num_transactions_last_hour": num_transactions_last_hour,
        "amount_vs_avg_ratio": np.round(amount_vs_avg_ratio, 2),
        "is_fraud": int(is_fraud),
    })


def _normal_hour_distribution():
    """Realistic hour distribution: peaks at lunch and evening."""
    probs = np.array([
        1, 0.5, 0.3, 0.2, 0.2, 0.3,  # 0-5 AM
        1, 2, 4, 5, 5, 6,              # 6-11 AM
        7, 6, 5, 5, 5, 6,              # 12-5 PM
        7, 8, 7, 5, 3, 2,              # 6-11 PM
    ], dtype=float)
    return probs / probs.sum()


def _fraud_hour_distribution():
    """Fraud peaks at late night / early morning."""
    probs = np.array([
        6, 7, 8, 8, 7, 5,  # 0-5 AM (peak fraud)
        3, 2, 2, 2, 2, 2,  # 6-11 AM
        2, 2, 2, 2, 3, 3,  # 12-5 PM
        4, 4, 4, 5, 6, 6,  # 6-11 PM
    ], dtype=float)
    return probs / probs.sum()


def _normal_merchant_distribution():
    """Normal: grocery and restaurant dominant."""
    probs = np.array([15, 12, 10, 8, 5, 6, 3, 5, 4, 8, 3, 5, 4, 7, 5], dtype=float)
    return probs / probs.sum()


def _fraud_merchant_distribution():
    """Fraud: online retail, electronics, ATM dominant."""
    probs = np.array([3, 2, 2, 18, 15, 5, 8, 3, 1, 1, 1, 2, 2, 5, 32], dtype=float)
    return probs / probs.sum()


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_fraud_data()
    output_path = get_project_root() / "data" / "raw" / "fraud_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions -> {output_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
