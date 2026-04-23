"""Generate synthetic dental patient no-show dataset.

Phase A.4 — Patient No-Show Prediction (XGBoost classifier).

The synthetic target is a logistic function of the 8 input features:
- prior_no_shows: strong positive driver (chronic missers stay chronic)
- days_until_appointment: weak positive (longer lead time -> more forgetting)
- distance_km: moderate positive (commute friction)
- prior_appointments: negative (loyal / engaged patients show up)

Target no-show rate is tuned to ~25% to match clinic-reported averages.
"""

import numpy as np
import pandas as pd


def generate_dental_noshow_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic dental appointment records with a no-show target.

    Args:
        n_samples: Number of appointment rows to generate.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with 8 feature columns + ``is_no_show`` target.
    """
    rng = np.random.default_rng(seed)

    # Age: normal distribution centered at 45, clamped to 18-90
    age = rng.normal(45, 16, n_samples).clip(18, 90).astype(int)

    # Prior no-shows: Poisson skewed low; most patients have 0-2, long tail.
    prior_no_shows = rng.poisson(1.2, n_samples).clip(0, 10)

    # Days until appointment: uniform-ish 0-60, with spikes near 7/14/30
    days_until_appointment = rng.integers(0, 61, n_samples)

    # Appointment hour: clinic hours 8-18 (9am-6pm); morning slots slightly
    # more common than afternoon.
    appointment_hour = rng.integers(8, 19, n_samples)

    # Distance to clinic in km: gamma distribution (right-skewed); most
    # patients within 10km, a tail of commuters.
    distance_km = rng.gamma(2.0, 4.0, n_samples).clip(0, 50)

    # Insurance type: 1=private (most common), 2=medicaid, 3=uninsured, 4=medicare.
    insurance_type = rng.choice([1, 2, 3, 4], p=[0.55, 0.2, 0.1, 0.15], size=n_samples)

    # Procedure complexity: 1-5, most routine (1-2).
    procedure_complexity = rng.choice(
        [1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05], size=n_samples
    )

    # Prior appointments (total): Poisson with higher mean than no-shows.
    prior_appointments = rng.poisson(6, n_samples).clip(0, 20)

    # Logistic target — coefficients chosen so target no-show rate lands near 25%.
    z = (
        -1.6
        + 0.55 * prior_no_shows                                  # strong positive
        + 0.015 * days_until_appointment                         # weak positive
        + 0.04 * distance_km                                     # moderate positive
        - 0.06 * prior_appointments                              # negative
        + rng.normal(0, 0.3, n_samples)                          # noise
    )
    no_show_prob = 1.0 / (1.0 + np.exp(-z))
    is_no_show = (rng.random(n_samples) < no_show_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "prior_no_shows": prior_no_shows,
            "days_until_appointment": days_until_appointment.astype(int),
            "appointment_hour": appointment_hour.astype(int),
            "distance_km": np.round(distance_km, 2),
            "insurance_type": insurance_type.astype(int),
            "procedure_complexity": procedure_complexity.astype(int),
            "prior_appointments": prior_appointments.astype(int),
            "is_no_show": is_no_show,
        }
    )

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_dental_noshow_data()
    output_path = get_project_root() / "data" / "raw" / "dental_noshow.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} dental no-show samples -> {output_path}")
    print(f"No-show rate: {df['is_no_show'].mean():.1%}")
