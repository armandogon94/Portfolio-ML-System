"""Generate synthetic H-1B visa approval dataset.

Inspired by DOL LCA (Labor Condition Application) disclosure data. All fields
are synthetic — no real applicant data is used. Feature relationships are
chosen so the classifier must learn a realistic multi-feature story (wage
level + employer history + education + seniority) rather than any single
dominating signal.

Target: ``is_approved`` (binary). Target approval rate ≈ 70% — matches the
rough ballpark of real LCA certifications without overfitting the agent to
any one fiscal year.
"""

import numpy as np
import pandas as pd


def generate_h1b_approval_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic H-1B petition data.

    Features (all shipped in the final DataFrame):

    - ``prevailing_wage`` (40k–300k): Strong positive signal above 80k
      (higher wage relative to SOC level ⇒ more credible specialty role).
    - ``soc_code_level`` (1–4; 4=senior specialty): Positive signal.
    - ``employer_size_tier`` (1–5): Large employers (4–5) trend positive.
    - ``job_level`` (1–4): Positive signal.
    - ``education_level`` (1–5; 1=Bachelor, 3=PhD, 4/5=PhD+add'l): Master's
      and above are positive.
    - ``experience_years`` (0–30): Positive.
    - ``country_of_citizenship_tier`` (1–5): Mild signal (backlog countries
      face more scrutiny downstream but LCA approval itself is mostly
      employer/role driven).
    - ``employer_prior_approval_rate`` (0–1): The strongest signal —
      employers with a track record of clean filings get approved.
    """
    rng = np.random.default_rng(seed)

    # Prevailing wage — log-normal so the long tail covers senior specialty
    # roles ($200k+) without dominating the distribution.
    prevailing_wage = rng.lognormal(mean=11.5, sigma=0.45, size=n_samples).clip(
        40_000, 300_000
    )

    # SOC code level — skewed toward mid-specialty (2–3)
    soc_code_level = rng.choice(
        [1, 2, 3, 4], size=n_samples, p=[0.10, 0.45, 0.35, 0.10]
    )

    # Employer size tier (1 = <50 employees, 5 = Fortune 500)
    employer_size_tier = rng.choice(
        [1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.20, 0.30, 0.25, 0.15]
    )

    # Job level — entry/mid/senior/principal
    job_level = rng.choice(
        [1, 2, 3, 4], size=n_samples, p=[0.20, 0.45, 0.25, 0.10]
    )

    # Education level (1=Bachelor, 2=Master, 3=PhD, 4=PhD+specialty,
    # 5=PhD+specialty+post-doc). Master/PhD concentrated because H-1B roles
    # typically require a specialty occupation.
    education_level = rng.choice(
        [1, 2, 3, 4, 5], size=n_samples, p=[0.30, 0.45, 0.15, 0.07, 0.03]
    )

    # Experience years — exponential, clipped.
    experience_years = rng.exponential(6, n_samples).clip(0, 30).astype(int)

    # Country tier (1=low-volume, 5=high-volume backlog country like IN/CN)
    country_of_citizenship_tier = rng.choice(
        [1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.20, 0.25, 0.20]
    )

    # Employer's prior approval rate — most sponsors are credible (beta
    # weighted toward 0.7–0.95), a long tail of dubious employers trends low.
    employer_prior_approval_rate = rng.beta(6, 2, n_samples).clip(0.0, 1.0)

    # Logistic target. Coefficients tuned empirically so overall approval
    # rate lands near 70% with realistic dispersion across feature values.
    # Intercept (positive) reflects the fact that LCA certifications are
    # approved more often than not; the features pull probabilities down
    # when the case is weak.
    z = (
        0.1
        # Prevailing wage: positive above 80k; clipping the z-score at ±2
        # keeps the tail from swamping other signals.
        + 0.9 * np.clip((prevailing_wage - 80_000) / 40_000, -2, 2)
        # SOC level and job level — seniority bumps the signal modestly.
        + 0.35 * (soc_code_level - 2)
        + 0.25 * (job_level - 2)
        # Larger employers a bit more likely (resources for clean filings).
        + 0.20 * (employer_size_tier - 3)
        # Master/PhD positive; bachelor alone neutral-to-negative.
        + 0.40 * np.where(education_level >= 3, 1.0, 0.0)
        + 0.20 * np.where(education_level == 2, 1.0, 0.0)
        # Experience — diminishing returns after ~10 yrs.
        + 0.05 * np.clip(experience_years, 0, 15)
        # Country tier: mild negative pressure for high-backlog countries.
        - 0.15 * (country_of_citizenship_tier - 3)
        # The dominant signal: employer's track record. Rescaled around 0.75
        # (the beta distribution's rough center) so a mediocre employer is
        # neutral and a bad-track-record one drags the whole case.
        + 4.0 * (employer_prior_approval_rate - 0.75)
        + rng.normal(0, 0.5, n_samples)  # noise
    )
    approval_prob = 1 / (1 + np.exp(-z))
    is_approved = (rng.random(n_samples) < approval_prob).astype(int)

    df = pd.DataFrame({
        "prevailing_wage": np.round(prevailing_wage, 2),
        "soc_code_level": soc_code_level.astype(int),
        "employer_size_tier": employer_size_tier.astype(int),
        "job_level": job_level.astype(int),
        "education_level": education_level.astype(int),
        "experience_years": experience_years.astype(int),
        "country_of_citizenship_tier": country_of_citizenship_tier.astype(int),
        "employer_prior_approval_rate": np.round(employer_prior_approval_rate, 3),
        "is_approved": is_approved,
    })

    return df


if __name__ == "__main__":
    from src.config import get_project_root

    df = generate_h1b_approval_data()
    output_path = get_project_root() / "data" / "raw" / "h1b_approval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} H-1B samples -> {output_path}")
    print(f"Approval rate: {df['is_approved'].mean():.1%}")
