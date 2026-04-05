"""Generate synthetic time series demand data with seasonality and trends."""

import numpy as np
import pandas as pd


PRODUCT_CATEGORIES = {
    "electronics": {"base_demand": 150, "trend_slope": 0.05, "seasonality_amp": 0.3},
    "clothing": {"base_demand": 200, "trend_slope": 0.02, "seasonality_amp": 0.4},
    "groceries": {"base_demand": 500, "trend_slope": 0.01, "seasonality_amp": 0.15},
    "furniture": {"base_demand": 50, "trend_slope": 0.03, "seasonality_amp": 0.25},
    "sports": {"base_demand": 100, "trend_slope": 0.04, "seasonality_amp": 0.5},
}


def generate_timeseries_data(n_years: int = 3, seed: int = 42) -> pd.DataFrame:
    """Generate daily demand data for multiple product categories.

    Components:
    - Linear trend (per product)
    - Yearly seasonality (sine wave, product-specific amplitude)
    - Weekly seasonality (weekend dip)
    - Holiday spikes
    - Gaussian noise
    """
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp("2022-01-01")
    n_days = n_years * 365
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    rows = []
    for product, params in PRODUCT_CATEGORIES.items():
        base = params["base_demand"]
        trend_slope = params["trend_slope"]
        season_amp = params["seasonality_amp"]

        for i, date in enumerate(dates):
            day_of_year = date.day_of_year
            day_of_week = date.day_of_week  # 0=Monday, 6=Sunday

            # Linear trend
            trend = base * (1 + trend_slope * i / 365)

            # Yearly seasonality (peaks in summer and holiday season)
            yearly = base * season_amp * (
                0.5 * np.sin(2 * np.pi * day_of_year / 365 - np.pi / 2)
                + 0.3 * np.sin(4 * np.pi * day_of_year / 365)
            )

            # Weekly seasonality (weekend dip for B2B, spike for B2C)
            if product in ("electronics", "clothing", "sports"):
                weekly = base * 0.15 if day_of_week >= 5 else 0
            else:
                weekly = -base * 0.1 if day_of_week >= 5 else 0

            # Holiday spikes
            is_holiday = _is_holiday(date)
            holiday_boost = base * 0.5 if is_holiday else 0

            # Noise
            noise = rng.normal(0, base * 0.08)

            demand = max(0, trend + yearly + weekly + holiday_boost + noise)

            rows.append({
                "date": date,
                "product_category": product,
                "demand": round(demand),
                "is_holiday": int(is_holiday),
            })

    df = pd.DataFrame(rows)
    return df


def _is_holiday(date: pd.Timestamp) -> bool:
    """Check if date falls on a US holiday (simplified)."""
    holidays = [
        (1, 1),    # New Year's Day
        (2, 14),   # Valentine's Day
        (7, 4),    # Independence Day
        (10, 31),  # Halloween
        (11, 25),  # Black Friday approx
        (11, 26),  # Thanksgiving approx
        (12, 25),  # Christmas
        (12, 26),  # Boxing Day
        (12, 31),  # New Year's Eve
    ]
    return (date.month, date.day) in holidays


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_timeseries_data()
    output_path = get_project_root() / "data" / "raw" / "daily_demand.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} daily demand records -> {output_path}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Products: {df['product_category'].nunique()}")
