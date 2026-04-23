"""Generate synthetic heart disease dataset (Cleveland-style).

Eight Cleveland-inspired features plus a binary `disease` target generated
via a logistic combination that mirrors well-known clinical risk factors:
age, male sex, asymptomatic chest pain (type 4), high cholesterol, elevated
ST depression (oldpeak), and exercise-induced angina all raise risk; a
higher max heart rate lowers risk. Baseline rate ~35%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_heart_disease_data(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Cleveland-style heart disease data.

    Features:
        age (int, 25-85)
        sex (int, 0=female, 1=male)
        chest_pain_type (int, 1-4; 4=asymptomatic = highest risk)
        resting_bp (int, 80-200 mmHg)
        cholesterol (int, 100-400 mg/dL)
        max_heart_rate (int, 60-220 bpm)
        exercise_angina (int, 0/1)
        oldpeak (float, 0-6; ST depression in mm)

    Target:
        disease (int, 0/1): synthesised via logistic combination of above.
    """
    rng = np.random.default_rng(seed)

    # Age: normal distribution centred at 54 (Cleveland mean ~54.4)
    age = rng.normal(54, 12, n_samples).clip(25, 85).astype(int)

    # Sex: ~68% male in Cleveland; use 0.55 to stay a bit more balanced.
    sex = rng.choice([0, 1], n_samples, p=[0.45, 0.55])

    # Chest pain type (1..4); 4 = asymptomatic carries highest risk.
    chest_pain_type = rng.choice([1, 2, 3, 4], n_samples, p=[0.10, 0.18, 0.25, 0.47])

    # Resting BP: normal ~130, heavier right tail.
    resting_bp = rng.normal(130, 18, n_samples).clip(80, 200).astype(int)

    # Cholesterol: log-normal-ish right-skew centred near 240.
    cholesterol = rng.normal(245, 55, n_samples).clip(100, 400).astype(int)

    # Max heart rate: weakly decreases with age (220 - age rule of thumb +
    # noise). Higher values imply healthier response, so they lower risk.
    max_heart_rate = (220 - age + rng.normal(0, 22, n_samples)).clip(60, 220).astype(int)

    # Exercise-induced angina: ~33% base rate.
    exercise_angina = rng.choice([0, 1], n_samples, p=[0.67, 0.33])

    # Oldpeak (ST depression): exponential, most near zero.
    oldpeak = rng.exponential(1.0, n_samples).clip(0, 6).round(2)

    # ── Target: logistic combination of clinical risk factors ─────────────
    #
    # Coefficient signs chosen so: age↑ risk, sex==1↑ risk, chest_pain_type==4
    # strongly↑ risk, cholesterol>240↑ risk, oldpeak↑ risk, max_heart_rate↓
    # risk, exercise_angina↑ risk. Intercept tuned for ~35% prevalence.
    z = (
        -3.0
        + 0.035 * (age - 54)
        + 0.7 * (sex == 1)
        + 1.5 * (chest_pain_type == 4)
        + 0.25 * (chest_pain_type == 3)
        + 0.008 * (resting_bp - 130)
        + 0.7 * ((cholesterol > 240).astype(int))
        + 0.5 * oldpeak
        - 0.025 * (max_heart_rate - 150)
        + 1.1 * exercise_angina
        + rng.normal(0, 0.3, n_samples)  # noise
    )
    disease_prob = 1 / (1 + np.exp(-z))
    disease = (rng.random(n_samples) < disease_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex.astype(int),
        "chest_pain_type": chest_pain_type.astype(int),
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina.astype(int),
        "oldpeak": oldpeak,
        "disease": disease,
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_heart_disease_data()
    output_path = get_project_root() / "data" / "raw" / "heart_disease.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} heart disease samples -> {output_path}")
    print(f"Disease rate: {df['disease'].mean():.1%}")
