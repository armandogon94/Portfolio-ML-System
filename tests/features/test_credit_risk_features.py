"""LendingClub feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import credit_risk_features


def test_is_deterministic(credit_risk_frame):
    first, _ = credit_risk_features.engineer_features(credit_risk_frame, fit=True)
    second, _ = credit_risk_features.engineer_features(credit_risk_frame, fit=True)
    pd.testing.assert_frame_equal(first, second)


def test_term_parses_the_leading_space_form():
    """LendingClub writes ' 36 months' with a leading space. It is not a typo."""
    frame = pd.DataFrame({"term": [" 36 months", " 60 months"]})
    engineered, _ = credit_risk_features.engineer_features(frame)
    assert engineered["term_months"].tolist() == [36.0, 60.0]


def test_grade_is_ordinal_not_nominal():
    frame = pd.DataFrame({"grade": ["A", "D", "G"]})
    engineered, _ = credit_risk_features.engineer_features(frame)
    assert engineered["grade_ordinal"].tolist() == [0.0, 3.0, 6.0]


def test_unknown_employment_length_is_nan_not_zero():
    """'n/a' means unknown. Coding it as 0 asserts 'no employment history'."""
    frame = pd.DataFrame({"emp_length": ["10+ years", "n/a", "< 1 year"]})
    engineered, _ = credit_risk_features.engineer_features(frame)
    values = engineered["emp_length_years"].tolist()
    assert values[0] == 10.0
    assert np.isnan(values[1])
    assert values[2] == 0.5


def test_ratios_do_not_divide_by_zero():
    frame = pd.DataFrame({"loan_amnt": [10_000.0], "annual_inc": [0.0], "installment": [300.0]})
    engineered, _ = credit_risk_features.engineer_features(frame)
    assert np.isnan(engineered["loan_to_income"].iloc[0])
    assert np.isnan(engineered["installment_to_income"].iloc[0])


def test_credit_history_uses_only_origination_dates(credit_risk_frame):
    engineered, _ = credit_risk_features.engineer_features(credit_risk_frame)
    history = engineered["credit_history_months"].dropna()
    assert len(history) > 0
    assert (history >= 0).all(), "a loan cannot precede the first credit line"


def test_split_key_is_not_a_feature(credit_risk_frame, credit_risk_config):
    engineered, _ = credit_risk_features.engineer_features(credit_risk_frame)
    selected = credit_risk_features.get_feature_columns(
        engineered, credit_risk_config["data"]["denylist"]
    )
    assert "issue_d" not in selected
    assert "issue_year" not in selected
    assert "fico_mid" in selected
