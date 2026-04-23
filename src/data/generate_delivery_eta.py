"""Generate synthetic delivery ETA dataset for the Logistics industry.

Each row represents a single parcel delivery; the target ``eta_hours`` is a
semi-linear function of distance, traffic, weather, origin/destination tier,
priority and a weekend bonus plus Gaussian noise. The generator mirrors the
deterministic style used by the other synthetic generators (credit risk,
housing) so the Phase A.7 pipeline stays self-contained.
"""

import numpy as np
import pandas as pd

# Penalty (hours) added to the ETA as routes move from hub↔hub (tier 1) out to
# remote↔remote (tier 4). Matches the heuristic in the Phase A.7 spec.
TIER_PENALTIES = {1: 0.0, 2: 1.0, 3: 3.0, 4: 6.0}


def generate_delivery_eta_data(n_samples: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic delivery parcel records with an ``eta_hours`` target.

    Features:
        - distance_km            float, 1–2000
        - package_weight_kg      float, 0.1–50
        - traffic_congestion     int, 1–5 (5 = peak)
        - weather_severity       int, 0–4 (0 = clear, 4 = storm)
        - time_of_day            int, 0–23 (hour of pickup)
        - day_of_week            int, 0–6 (Mon=0, Sun=6)
        - carrier_priority       int, 1–3 (1 = standard, 3 = express)
        - origin_destination_tier int, 1–4 (1 = hub↔hub, 4 = remote↔remote)

    Target:
        eta_hours = distance_km / 80
                  + 0.5 * traffic_congestion
                  + 0.3 * weather_severity
                  + tier_penalty
                  - 0.8 * (carrier_priority - 1)
                  + weekend_bonus (0.5h on Sat/Sun)
                  + N(0, 2)
        clipped to [2, 120].
    """
    rng = np.random.default_rng(seed)

    # Distance — log-normal-ish so local and long-haul shipments both appear.
    distance_km = rng.lognormal(mean=4.0, sigma=0.9, size=n_samples).clip(1, 2000).round(1)

    # Package weight — right-skewed, small parcels are most common.
    package_weight_kg = rng.lognormal(
        mean=0.6, sigma=1.0, size=n_samples
    ).clip(0.1, 50).round(2)

    # Traffic / weather — categorical scales skewed toward typical conditions.
    traffic_congestion = rng.choice(
        [1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.30, 0.30, 0.15, 0.10]
    )
    weather_severity = rng.choice(
        [0, 1, 2, 3, 4], size=n_samples, p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )

    # Time of pickup — near-uniform over business hours, sparse overnight.
    time_of_day = rng.integers(0, 24, size=n_samples)

    # Day of week — slightly more weekday shipments.
    day_of_week = rng.choice(
        [0, 1, 2, 3, 4, 5, 6],
        size=n_samples,
        p=[0.17, 0.17, 0.17, 0.17, 0.17, 0.08, 0.07],
    )

    carrier_priority = rng.choice([1, 2, 3], size=n_samples, p=[0.50, 0.35, 0.15])
    origin_destination_tier = rng.choice(
        [1, 2, 3, 4], size=n_samples, p=[0.30, 0.35, 0.25, 0.10]
    )

    tier_penalty = np.array([TIER_PENALTIES[int(t)] for t in origin_destination_tier])
    weekend_bonus = np.where(day_of_week >= 5, 0.5, 0.0)

    base_eta = (
        distance_km / 80.0
        + 0.5 * traffic_congestion
        + 0.3 * weather_severity
        + tier_penalty
        - 0.8 * (carrier_priority - 1)
        + weekend_bonus
        + rng.normal(0.0, 2.0, size=n_samples)
    )
    eta_hours = np.clip(base_eta, 2.0, 120.0).round(2)

    df = pd.DataFrame({
        "distance_km": distance_km,
        "package_weight_kg": package_weight_kg,
        "traffic_congestion": traffic_congestion.astype(int),
        "weather_severity": weather_severity.astype(int),
        "time_of_day": time_of_day.astype(int),
        "day_of_week": day_of_week.astype(int),
        "carrier_priority": carrier_priority.astype(int),
        "origin_destination_tier": origin_destination_tier.astype(int),
        "eta_hours": eta_hours,
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_delivery_eta_data()
    output_path = get_project_root() / "data" / "raw" / "delivery_eta.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} delivery ETA records -> {output_path}")
    print(f"ETA range: {df['eta_hours'].min():.1f} - {df['eta_hours'].max():.1f} hours")
    print(f"Median ETA: {df['eta_hours'].median():.1f} hours")
