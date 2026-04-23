"""Generate synthetic rental-price dataset for the Real Estate industry (A.3).

Target: ``nightly_rate`` (USD) — a regression task. Features are generated with
realistic correlations so a LightGBM regressor can learn the underlying formula:

    base = 40 * square_feet / 100
         + 30 * bedrooms
         + 20 * bathrooms
         + 25 * amenity_score
         + location_multiplier (1.0 at tier 1 -> 2.5 at tier 5)
         - 2 * distance_to_downtown_km
         + 0.6 * peer_nightly_rate
         + noise ~ N(0, 30)

Clipped to [50, 800] so the target range matches the spec.
"""

import numpy as np
import pandas as pd

# Location tier multiplier: linear ramp from 1.0 (tier 1) to 2.5 (tier 5).
_LOCATION_MULTIPLIER = {1: 1.0, 2: 1.375, 3: 1.75, 4: 2.125, 5: 2.5}


def generate_rental_price_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic rental listing data for nightly-rate prediction.

    Args:
        n_samples: Number of listings to generate.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with 8 input features plus ``nightly_rate`` target.
    """
    rng = np.random.default_rng(seed)

    # Bedrooms: skewed toward 1-3, capped at 6.
    bedrooms = rng.choice([0, 1, 2, 3, 4, 5, 6], size=n_samples,
                          p=[0.05, 0.25, 0.30, 0.20, 0.12, 0.06, 0.02])

    # Bathrooms: typically 1-2, rarely 3-4.
    bathrooms = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.55, 0.30, 0.12, 0.03])

    # Square feet: correlated with bedrooms (roughly 400 sqft base + 300 per bed).
    base_sqft = 400 + bedrooms * 300 + rng.normal(0, 150, n_samples)
    square_feet = base_sqft.clip(300, 5000).astype(int)

    # Property type: 1=studio, 2=apartment, 3=house, 4=condo.
    property_type = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.10, 0.45, 0.25, 0.20])

    # Location tier: 1-5 (5 most desirable). Uniform-ish with slight bias toward middle.
    location_tier = rng.choice([1, 2, 3, 4, 5], size=n_samples,
                               p=[0.15, 0.25, 0.30, 0.20, 0.10])

    # Distance to downtown: exponential, clipped to 50km.
    distance_to_downtown_km = rng.exponential(8, n_samples).clip(0, 50)

    # Amenity score: 0-10, slight skew toward middle values.
    amenity_score = rng.integers(0, 11, size=n_samples)

    # Peer nightly rate: log-normal around $150 nightly.
    peer_nightly_rate = rng.lognormal(mean=4.9, sigma=0.4, size=n_samples).clip(30, 600)

    # Build target from the declared synthetic formula.
    location_mult = np.array([_LOCATION_MULTIPLIER[t] for t in location_tier])
    base = (
        40 * square_feet / 100
        + 30 * bedrooms
        + 20 * bathrooms
        + 25 * amenity_score
        + location_mult * 30  # scale multiplier so it meaningfully shifts price
        - 2 * distance_to_downtown_km
        + 0.6 * peer_nightly_rate
        + rng.normal(0, 30, n_samples)
    )
    nightly_rate = np.clip(base, 50, 800)

    df = pd.DataFrame({
        "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int),
        "square_feet": square_feet,
        "property_type": property_type.astype(int),
        "location_tier": location_tier.astype(int),
        "distance_to_downtown_km": np.round(distance_to_downtown_km, 2),
        "amenity_score": amenity_score.astype(int),
        "peer_nightly_rate": np.round(peer_nightly_rate, 2),
        "nightly_rate": np.round(nightly_rate, 2),
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_rental_price_data()
    output_dir = get_project_root() / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rental_price.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rental price samples -> {output_path}")
    print(
        f"Nightly rate mean=${df['nightly_rate'].mean():.2f}, "
        f"min=${df['nightly_rate'].min():.2f}, max=${df['nightly_rate'].max():.2f}"
    )
