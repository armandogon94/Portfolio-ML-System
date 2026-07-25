"""Feature engineering for LendingClub consumer credit risk.

Same contract as ``fraud_features``: one implementation, shared by training and
serving. Only origination-time information is used. The denylist in
``configs/credit_risk.yaml`` is the hard boundary; this module additionally never
constructs a feature from a post-origination column even if one leaks into the
frame, because it names its inputs explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: LendingClub grades are ordinal, not nominal. Encoding them as an integer keeps
#: the ordering the trees would otherwise have to rediscover from one-hots.
GRADE_ORDER = {letter: i for i, letter in enumerate("ABCDEFG")}

#: ``emp_length`` arrives as free text. Mapped to years; "n/a" becomes NaN rather
#: than 0, because "we do not know" and "no employment history" differ.
EMP_LENGTH_YEARS = {
    "< 1 year": 0.5,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


def engineer_features(
    frame: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add engineered credit-risk features.

    Args:
        frame: Canonical LendingClub frame.
        artifacts: Unused for this problem — kept so all three feature modules
            share one signature and ``tabular.py`` needs no branching.
        fit: Unused, same reason.

    Returns:
        ``(engineered_frame, artifacts)``.
    """
    del fit  # this problem has no fitted feature state
    out = frame.copy()
    artifacts = dict(artifacts or {})

    if "term" in out.columns:
        # " 36 months" -> 36. The leading space is in the source data.
        out["term_months"] = out["term"].astype("object").str.extract(r"(\d+)")[0].astype("float32")

    if "grade" in out.columns:
        out["grade_ordinal"] = out["grade"].astype("object").map(GRADE_ORDER).astype("float32")

    if "sub_grade" in out.columns:
        sub = out["sub_grade"].astype("object")
        out["sub_grade_ordinal"] = sub.str[0].map(GRADE_ORDER).astype(
            "float32"
        ) * 5 + pd.to_numeric(sub.str[1:], errors="coerce").astype("float32")

    if "emp_length" in out.columns:
        out["emp_length_years"] = (
            out["emp_length"].astype("object").map(EMP_LENGTH_YEARS).astype("float32")
        )

    if "fico_range_low" in out.columns and "fico_range_high" in out.columns:
        out["fico_mid"] = ((out["fico_range_low"] + out["fico_range_high"]) / 2).astype("float32")

    if {"loan_amnt", "annual_inc"}.issubset(out.columns):
        # Loan-to-income: the single most interpretable affordability ratio, and
        # the one a credit analyst will ask about first.
        out["loan_to_income"] = (out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)).astype(
            "float32"
        )

    if {"installment", "annual_inc"}.issubset(out.columns):
        monthly_income = (out["annual_inc"] / 12).replace(0, np.nan)
        out["installment_to_income"] = (out["installment"] / monthly_income).astype("float32")

    if {"revol_bal", "annual_inc"}.issubset(out.columns):
        out["revol_to_income"] = (out["revol_bal"] / out["annual_inc"].replace(0, np.nan)).astype(
            "float32"
        )

    if {"issue_d", "earliest_cr_line"}.issubset(out.columns):
        # Credit-history length at origination, in months. Derived from two
        # origination-time dates, so it is not leakage.
        months = (out["issue_d"] - out["earliest_cr_line"]).dt.days / 30.44
        out["credit_history_months"] = months.astype("float32")

    if "issue_d" in out.columns:
        # Vintage year is a legitimate feature (macro conditions differ by
        # cohort) but it is also the split key, so it is excluded from the
        # feature list below. Kept here only for the split and for reporting.
        out["issue_year"] = out["issue_d"].dt.year.astype("float32")

    return out, artifacts


#: Columns present in the frame purely for splitting, reporting or as raw text
#: that has been replaced by an encoded version.
_STRUCTURAL = {
    "issue_d",
    "earliest_cr_line",
    "issue_year",
    "term",
    "grade",
    "sub_grade",
    "emp_length",
}


def get_feature_columns(frame: pd.DataFrame, denylist: list[str] | None = None) -> list[str]:
    """Return the ordered feature columns for an engineered frame."""
    blocked = set(denylist or []) | _STRUCTURAL
    return sorted(c for c in frame.columns if c not in blocked)
