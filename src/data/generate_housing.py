"""Generate synthetic housing dataset with realistic feature correlations."""

import numpy as np
import pandas as pd


NEIGHBORHOOD_TIERS = {
    1: {"name": "budget", "multiplier": 0.65},
    2: {"name": "affordable", "multiplier": 0.80},
    3: {"name": "moderate", "multiplier": 1.00},
    4: {"name": "upscale", "multiplier": 1.35},
    5: {"name": "luxury", "multiplier": 1.80},
}


def generate_housing_data(n_samples: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic real estate data.

    Price is a polynomial function of features with neighborhood multipliers
    and realistic Gaussian noise. Features have natural correlations
    (e.g., bedrooms correlated with square footage).
    """
    rng = np.random.default_rng(seed)

    # Square footage: normal distribution
    square_feet = rng.normal(1800, 600, n_samples).clip(400, 8000).astype(int)

    # Bedrooms: correlated with square footage
    bedrooms_base = (square_feet / 500).clip(1, 6)
    bedrooms = (bedrooms_base + rng.normal(0, 0.5, n_samples)).clip(1, 6).astype(int)

    # Bathrooms: correlated with bedrooms
    bathrooms = (bedrooms * 0.7 + rng.normal(0, 0.3, n_samples)).clip(1, 5).round(0).astype(int)

    # Year built: uniform 1950-2024
    year_built = rng.integers(1950, 2025, n_samples)

    # Lot size: correlated with square footage, larger variance
    lot_size_sqft = (square_feet * rng.uniform(2, 8, n_samples)).clip(1000, 50000).astype(int)

    # Garage spaces
    garage_spaces = rng.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.45, 0.15])

    # Has pool
    has_pool = rng.binomial(1, 0.15, n_samples)

    # Neighborhood tier
    neighborhood_tier = rng.choice([1, 2, 3, 4, 5], n_samples, p=[0.10, 0.20, 0.35, 0.25, 0.10])

    # Proximity to city center (miles)
    proximity_to_city_center = rng.exponential(10, n_samples).clip(0.5, 50).round(1)

    # Generate price as polynomial function of features
    base_price = (
        50 * square_feet
        + 15000 * bedrooms
        + 20000 * bathrooms
        + 500 * (year_built - 1950)
        + 2 * lot_size_sqft
        + 25000 * garage_spaces
        + 40000 * has_pool
        - 3000 * proximity_to_city_center
    )

    # Apply neighborhood multiplier
    multipliers = np.array([NEIGHBORHOOD_TIERS[t]["multiplier"] for t in neighborhood_tier])
    price = base_price * multipliers

    # Add realistic noise (10% standard deviation)
    noise = rng.normal(1.0, 0.10, n_samples)
    price = (price * noise).clip(50000, 3000000).round(-2)

    df = pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "lot_size_sqft": lot_size_sqft,
        "garage_spaces": garage_spaces,
        "has_pool": has_pool,
        "neighborhood_tier": neighborhood_tier,
        "proximity_to_city_center": proximity_to_city_center,
        "price": price.astype(int),
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_housing_data()
    output_path = get_project_root() / "data" / "raw" / "housing.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} housing records -> {output_path}")
    print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"Median price: ${df['price'].median():,.0f}")
