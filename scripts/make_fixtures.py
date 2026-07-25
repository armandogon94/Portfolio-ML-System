#!/usr/bin/env python
"""Regenerate the committed CI fixtures in ``data/sample/``.

These fixtures are **schema-shaped synthetic noise**. Read that twice: they exist
only so that CI, ``make test`` and ``tests/e2e/test_train_to_serve.py`` can run
without credentials and without touching a real dataset.

They are NOT training data and no number computed from them is ever published.
The guard is structural, not a promise: ``src/training/tabular.py`` refuses to
write a checkpoint when ``--sample`` is set, and ``tests/test_quality_gates.py``
skips rather than passes when no real checkpoint exists.

The CSVs contain only their schema header and data rows. Their synthetic status
is documented in ``data/README.md`` rather than written as a comment row because
the adapters read them with the standard CSV header parser.

Why synthetic fixtures at all, when the whole point of this rebuild was to stop
using synthetic data? Because IEEE-CIS competition data is not redistributable —
committing 500 real rows would violate the competition rules. This is the one
place synthetic data survives, and it survives for a licensing reason, not a
convenience one. See docs/adr/0003-real-data-over-synthetic.md.

Usage:
    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import get_project_root
from src.data.adapters import credit_card_churn, ieee_cis, lending_club

SEED = 42
N_ROWS = 500


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def make_ieee_cis_fixture() -> pd.DataFrame:
    """A 500-row frame with IEEE-CIS's column names and plausible dtypes.

    The label is drawn INDEPENDENTLY of the features — a Bernoulli(0.035) draw
    with no relationship to any column. That is deliberate and it is the opposite
    of what the deleted ``generate_fraud.py`` did. A model trained on this fixture
    should score ~0.5 AUC. If a test ever reports a good score here, the test is
    broken.
    """
    rng = _rng()
    columns: dict[str, np.ndarray] = {
        "TransactionID": np.arange(2_987_000, 2_987_000 + N_ROWS, dtype=np.int64),
        "isFraud": rng.binomial(1, 0.035, N_ROWS).astype(np.int8),
        "TransactionDT": np.sort(rng.integers(86_400, 15_811_131, N_ROWS)).astype(np.int64),
        "TransactionAmt": np.round(rng.lognormal(4.0, 1.1, N_ROWS), 2).astype(np.float32),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], N_ROWS),
        "card1": rng.integers(1000, 18_400, N_ROWS).astype(np.float32),
        "card2": rng.integers(100, 600, N_ROWS).astype(np.float32),
        "card3": rng.choice([150.0, 185.0], N_ROWS).astype(np.float32),
        "card4": rng.choice(["visa", "mastercard", "discover", "american express"], N_ROWS),
        "card5": rng.integers(100, 240, N_ROWS).astype(np.float32),
        "card6": rng.choice(["debit", "credit"], N_ROWS),
        "addr1": rng.integers(100, 540, N_ROWS).astype(np.float32),
        "addr2": np.full(N_ROWS, 87.0, dtype=np.float32),
        "dist1": rng.exponential(30, N_ROWS).astype(np.float32),
        "dist2": rng.exponential(120, N_ROWS).astype(np.float32),
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com"], N_ROWS
        ),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], N_ROWS),
    }
    for name in (f"C{i}" for i in range(1, 15)):
        columns[name] = rng.integers(0, 40, N_ROWS).astype(np.float32)
    for name in (f"D{i}" for i in range(1, 16)):
        columns[name] = rng.integers(0, 600, N_ROWS).astype(np.float32)
    for name in (f"M{i}" for i in range(1, 10)):
        columns[name] = rng.choice(["T", "F"], N_ROWS)
    # 40 of the 339 V columns is enough to exercise the wide-frame code paths
    # without a 500x339 CSV in git.
    for name in (f"V{i}" for i in range(1, 41)):
        columns[name] = rng.integers(0, 8, N_ROWS).astype(np.float32)

    frame = pd.DataFrame(columns)
    # Real IEEE-CIS is heavily missing. Reproduce that so the fixture exercises
    # LightGBM's native NaN handling instead of a dense happy path.
    for name in ("dist2", "card2", "D3", "D4", "V7"):
        mask = rng.random(N_ROWS) < 0.4
        frame.loc[mask, name] = np.nan
    return frame


def make_lending_club_fixture() -> pd.DataFrame:
    """A 500-row LendingClub-shaped frame. ``loan_status`` is independent noise."""
    rng = _rng()
    issue_dates = pd.to_datetime(
        rng.choice(pd.date_range("2012-01-01", "2018-10-01", freq="MS"), N_ROWS)
    )
    frame = pd.DataFrame(
        {
            # A third of rows are non-terminal on purpose so the adapter's
            # filtering logic is actually exercised by the fixture.
            "loan_status": rng.choice(
                ["Fully Paid", "Charged Off", "Current"], N_ROWS, p=[0.55, 0.15, 0.30]
            ),
            "issue_d": issue_dates.strftime("%b-%Y"),
            "loan_amnt": rng.choice(np.arange(1000, 40_001, 500), N_ROWS).astype(float),
            "funded_amnt": rng.choice(np.arange(1000, 40_001, 500), N_ROWS).astype(float),
            "term": rng.choice([" 36 months", " 60 months"], N_ROWS),
            "int_rate": np.round(rng.uniform(5.3, 30.9, N_ROWS), 2),
            "installment": np.round(rng.uniform(30, 1500, N_ROWS), 2),
            "grade": rng.choice(list("ABCDEFG"), N_ROWS),
            "sub_grade": [
                f"{g}{n}"
                for g, n in zip(rng.choice(list("ABCDEFG"), N_ROWS), rng.integers(1, 6, N_ROWS))
            ],
            "emp_length": rng.choice(
                ["< 1 year", "1 year", "3 years", "5 years", "10+ years"], N_ROWS
            ),
            "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], N_ROWS),
            "annual_inc": np.round(rng.lognormal(11.0, 0.6, N_ROWS), 2),
            "verification_status": rng.choice(
                ["Verified", "Source Verified", "Not Verified"], N_ROWS
            ),
            "purpose": rng.choice(
                ["debt_consolidation", "credit_card", "home_improvement", "other"], N_ROWS
            ),
            "addr_state": rng.choice(["CA", "NY", "TX", "FL", "IL"], N_ROWS),
            "dti": np.round(rng.uniform(0, 40, N_ROWS), 2),
            "delinq_2yrs": rng.integers(0, 4, N_ROWS).astype(float),
            "earliest_cr_line": pd.to_datetime(
                rng.choice(pd.date_range("1990-01-01", "2010-01-01", freq="MS"), N_ROWS)
            ).strftime("%b-%Y"),
            "fico_range_low": rng.choice(np.arange(660, 845, 5), N_ROWS).astype(float),
            "inq_last_6mths": rng.integers(0, 6, N_ROWS).astype(float),
            "mths_since_last_delinq": rng.integers(0, 120, N_ROWS).astype(float),
            "open_acc": rng.integers(2, 30, N_ROWS).astype(float),
            "pub_rec": rng.integers(0, 3, N_ROWS).astype(float),
            "revol_bal": np.round(rng.uniform(0, 60_000, N_ROWS), 2),
            "revol_util": np.round(rng.uniform(0, 100, N_ROWS), 1),
            "total_acc": rng.integers(4, 60, N_ROWS).astype(float),
            "application_type": rng.choice(["Individual", "Joint App"], N_ROWS),
            "mort_acc": rng.integers(0, 5, N_ROWS).astype(float),
            "pub_rec_bankruptcies": rng.integers(0, 2, N_ROWS).astype(float),
        }
    )
    frame["fico_range_high"] = frame["fico_range_low"] + 4
    return frame


def make_churn_fixture() -> pd.DataFrame:
    """A 500-row attrition-shaped frame, including the two Naive-Bayes leak columns.

    The leak columns are present so ``tests/data/test_leakage_denylist.py`` has
    something real to assert against — the test proves the denylist removes them.
    """
    rng = _rng()
    attrited = rng.binomial(1, 0.1607, N_ROWS)
    frame = pd.DataFrame(
        {
            "CLIENTNUM": np.arange(768_800_000, 768_800_000 + N_ROWS, dtype=np.int64),
            "Attrition_Flag": np.where(attrited == 1, "Attrited Customer", "Existing Customer"),
            "Customer_Age": rng.integers(26, 74, N_ROWS).astype(float),
            "Gender": rng.choice(["M", "F"], N_ROWS),
            "Dependent_count": rng.integers(0, 6, N_ROWS).astype(float),
            "Education_Level": rng.choice(
                ["High School", "Graduate", "Uneducated", "College", "Doctorate", "Unknown"],
                N_ROWS,
            ),
            "Marital_Status": rng.choice(["Married", "Single", "Divorced", "Unknown"], N_ROWS),
            "Income_Category": rng.choice(
                ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
                N_ROWS,
            ),
            "Card_Category": rng.choice(["Blue", "Silver", "Gold", "Platinum"], N_ROWS),
            "Months_on_book": rng.integers(13, 57, N_ROWS).astype(float),
            "Total_Relationship_Count": rng.integers(1, 7, N_ROWS).astype(float),
            "Months_Inactive_12_mon": rng.integers(0, 7, N_ROWS).astype(float),
            "Contacts_Count_12_mon": rng.integers(0, 7, N_ROWS).astype(float),
            "Credit_Limit": np.round(rng.uniform(1438, 34_516, N_ROWS), 1),
            "Total_Revolving_Bal": rng.integers(0, 2518, N_ROWS).astype(float),
            "Total_Amt_Chng_Q4_Q1": np.round(rng.uniform(0, 3.4, N_ROWS), 3),
            "Total_Trans_Amt": rng.integers(510, 18_485, N_ROWS).astype(float),
            "Total_Trans_Ct": rng.integers(10, 140, N_ROWS).astype(float),
            "Total_Ct_Chng_Q4_Q1": np.round(rng.uniform(0, 3.7, N_ROWS), 3),
            "Avg_Utilization_Ratio": np.round(rng.uniform(0, 1, N_ROWS), 3),
        }
    )
    frame["Avg_Open_To_Buy"] = (frame["Credit_Limit"] - frame["Total_Revolving_Bal"]).round(1)
    # Mirror the real file's two pre-computed posteriors so the leakage
    # demonstration has real columns to remove.
    frame[credit_card_churn.NAIVE_BAYES_LEAK_COLUMNS[0]] = np.where(
        attrited == 1, rng.uniform(0.0, 0.2, N_ROWS), rng.uniform(0.8, 1.0, N_ROWS)
    ).round(6)
    frame[credit_card_churn.NAIVE_BAYES_LEAK_COLUMNS[1]] = (
        1 - frame[credit_card_churn.NAIVE_BAYES_LEAK_COLUMNS[0]]
    ).round(6)
    return frame


BUILDERS = {
    "ieee_cis_sample.csv": (make_ieee_cis_fixture, ieee_cis),
    "lending_club_sample.csv": (make_lending_club_fixture, lending_club),
    "churn_sample.csv": (make_churn_fixture, credit_card_churn),
}


def main() -> int:
    out_dir = get_project_root() / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, (builder, adapter) in BUILDERS.items():
        frame = builder()
        path = out_dir / filename
        frame.to_csv(path, index=False)

        # Round-trip through the adapter so a broken fixture fails here, not in CI.
        loaded = adapter.load(sample=True)
        print(
            f"{filename:26s} {len(frame):4d} rows -> adapter yields "
            f"{loaded.shape[0]:4d} x {loaded.shape[1]:3d}, "
            f"positive rate {loaded[adapter.TARGET].mean():.4f}  "
            f"({path.stat().st_size / 1024:.0f} KB)"
        )
    print("\nAll fixtures regenerated. Their synthetic status is documented in data/README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
