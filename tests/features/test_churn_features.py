"""Attrition feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import churn_features


def test_is_deterministic(churn_frame):
    first, _ = churn_features.engineer_features(churn_frame, fit=True)
    second, _ = churn_features.engineer_features(churn_frame, fit=True)
    pd.testing.assert_frame_equal(first, second)


def test_average_ticket_size():
    frame = pd.DataFrame({"Total_Trans_Amt": [1000.0, 500.0], "Total_Trans_Ct": [10.0, 0.0]})
    engineered, _ = churn_features.engineer_features(frame)
    assert engineered["avg_transaction_value"].iloc[0] == 100.0
    assert np.isnan(engineered["avg_transaction_value"].iloc[1])


def test_utilisation_and_contact_ratios_exist(churn_frame):
    engineered, _ = churn_features.engineer_features(churn_frame)
    for column in (
        "revolving_utilisation",
        "inactive_share_of_tenure",
        "contacts_per_product",
    ):
        assert column in engineered.columns


def test_engineered_columns_do_not_encode_the_target(churn_frame):
    engineered, _ = churn_features.engineer_features(churn_frame)
    target = engineered["is_attrited"].astype(float)
    for column in set(engineered.columns) - set(churn_frame.columns):
        values = pd.to_numeric(engineered[column], errors="coerce")
        if values.notna().sum() < 10 or values.nunique() < 2:
            continue
        assert abs(values.corr(target)) < 0.99, column
