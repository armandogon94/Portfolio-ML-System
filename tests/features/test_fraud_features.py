"""Fraud feature engineering: determinism, no target leakage, train-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import fraud_features


def test_is_deterministic(fraud_frame):
    first, _ = fraud_features.engineer_features(fraud_frame, fit=True)
    second, _ = fraud_features.engineer_features(fraud_frame, fit=True)
    pd.testing.assert_frame_equal(first, second)


def test_engineered_columns_never_reference_the_target(fraud_frame):
    """No engineered column may correlate perfectly with isFraud.

    A perfect correlation on a fixture whose label is independent noise can only
    come from the label having been used to build the feature.
    """
    engineered, _ = fraud_features.engineer_features(fraud_frame, fit=True)
    target = engineered["isFraud"].astype(float)
    new_columns = set(engineered.columns) - set(fraud_frame.columns)

    for column in new_columns:
        values = pd.to_numeric(engineered[column], errors="coerce")
        if values.notna().sum() < 10 or values.nunique() < 2:
            continue
        correlation = abs(values.corr(target))
        assert correlation < 0.99, f"{column} is almost identical to the target"


def test_fit_false_without_artifacts_is_an_error(fraud_frame):
    """Silently emitting all-NaN frequency features would degrade the model quietly."""
    with pytest.raises(ValueError, match="requires the artifacts"):
        fraud_features.engineer_features(fraud_frame, None, fit=False)


def test_frequency_maps_come_from_the_fit_split_only(fraud_frame):
    """A card seen only in test must map to NaN, not to its test-set count.

    This is the subtle leak: fitting frequency encodings on the whole frame lets
    the model know how often a card appears in the future.
    """
    train, test = fraud_frame.iloc[:300], fraud_frame.iloc[300:].copy()
    _, artifacts = fraud_features.engineer_features(train, fit=True)

    unseen = 999_999
    assert unseen not in set(train["card1"].dropna())
    test.loc[test.index[0], "card1"] = unseen

    engineered, _ = fraud_features.engineer_features(test, artifacts, fit=False)
    assert np.isnan(engineered["card1_freq"].iloc[0])


def test_amount_decimal_captures_the_cents_part():
    frame = pd.DataFrame(
        {
            "TransactionAmt": [100.00, 59.99, 12.345],
            "TransactionDT": [86_400, 90_000, 100_000],
            "isFraud": [0, 0, 1],
        }
    )
    engineered, _ = fraud_features.engineer_features(frame, fit=True)
    assert engineered["amt_decimal"].tolist() == [0.0, 990.0, 345.0]
    assert engineered["amt_is_round"].tolist() == [1, 0, 0]


def test_raw_identifiers_are_excluded_from_the_feature_list(fraud_frame, fraud_config):
    """card1 itself must not be a feature; card1_freq must be."""
    engineered, _ = fraud_features.engineer_features(fraud_frame, fit=True)
    selected = fraud_features.get_feature_columns(engineered, fraud_config["data"]["denylist"])
    assert "card1" not in selected
    assert "card1_freq" in selected
    assert "tx_day" not in selected, "the bare day index is the split boundary"


def test_d_columns_are_detrended(fraud_frame):
    engineered, _ = fraud_features.engineer_features(fraud_frame, fit=True)
    assert "D1_detrend" in engineered.columns
    expected = engineered["D1"].astype("float32") - engineered["tx_day"].astype("float32")
    pd.testing.assert_series_equal(
        engineered["D1_detrend"], expected.astype("float32"), check_names=False
    )
