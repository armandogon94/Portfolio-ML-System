"""Splits. The time-based ones must never put the future in train.

A random split on IEEE-CIS puts the same card on both sides of the boundary; on
LendingClub it mixes 2012 and 2018 vintages. Both inflate the score by several
points and neither survives deployment. These tests are the guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import make_splits, stratified_kfold_indices, time_split


@pytest.fixture
def timed_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "t": rng.permutation(np.arange(1000)),  # shuffled on purpose
            "y": rng.binomial(1, 0.2, 1000),
        }
    )


def test_no_train_timestamp_is_later_than_the_earliest_test_timestamp(timed_frame):
    split = time_split(timed_frame, "t", test_size=0.2, val_size=0.1)
    train_max = timed_frame["t"].to_numpy()[split.train].max()
    test_min = timed_frame["t"].to_numpy()[split.test].min()
    assert train_max < test_min


def test_validation_sits_between_train_and_test_in_time(timed_frame):
    """Early stopping on a validation fold from the future is still leakage."""
    split = time_split(timed_frame, "t", test_size=0.2, val_size=0.1)
    times = timed_frame["t"].to_numpy()
    assert times[split.train].max() < times[split.val].min()
    assert times[split.val].max() < times[split.test].min()


def test_the_three_partitions_are_disjoint_and_complete(timed_frame):
    split = time_split(timed_frame, "t", test_size=0.2, val_size=0.1)
    combined = np.concatenate([split.train, split.val, split.test])
    assert len(combined) == len(timed_frame)
    assert len(set(combined)) == len(timed_frame)


def test_split_sizes_match_the_requested_fractions(timed_frame):
    split = time_split(timed_frame, "t", test_size=0.2, val_size=0.1)
    assert len(split.test) == 200
    assert len(split.val) == 100
    assert len(split.train) == 700


def test_repeated_timestamps_never_span_train_validation_or_test():
    """Monthly timestamps must move as a group to the earlier partition."""
    months = np.repeat(pd.period_range("2017-01", periods=12, freq="M"), 30)
    frame = pd.DataFrame({"issue_d": months.to_timestamp(), "y": np.tile([0, 1], 180)})

    split = time_split(frame, "issue_d", test_size=0.2, val_size=0.1)
    timestamp_sets = [
        set(frame.iloc[indices]["issue_d"]) for indices in (split.train, split.val, split.test)
    ]

    assert timestamp_sets[0].isdisjoint(timestamp_sets[1])
    assert timestamp_sets[0].isdisjoint(timestamp_sets[2])
    assert timestamp_sets[1].isdisjoint(timestamp_sets[2])
    assert frame.iloc[split.train]["issue_d"].max() < frame.iloc[split.val]["issue_d"].min()
    assert frame.iloc[split.val]["issue_d"].max() < frame.iloc[split.test]["issue_d"].min()


def test_tie_safe_boundary_refuses_to_empty_the_later_partition():
    frame = pd.DataFrame({"t": [1] * 7 + [2] * 3, "y": [0, 1] * 5})

    with pytest.raises(ValueError, match="timestamp ties.*test partition empty"):
        time_split(frame, "t", test_size=0.2, val_size=0.0)


def test_missing_split_column_names_the_columns_it_did_find(timed_frame):
    with pytest.raises(KeyError, match="issue_d"):
        time_split(timed_frame, "issue_d")


def test_impossible_fractions_are_refused(timed_frame):
    with pytest.raises(ValueError, match="must be < 1.0"):
        time_split(timed_frame, "t", test_size=0.8, val_size=0.3)


def test_stratified_kfold_preserves_the_class_balance():
    rng = np.random.default_rng(3)
    labels = rng.binomial(1, 0.16, 1000)
    folds = stratified_kfold_indices(labels, n_splits=5, seed=42)

    assert len(folds) == 5
    overall = labels.mean()
    for fold in folds:
        assert abs(labels[fold.test].mean() - overall) < 0.03


def test_stratified_kfold_test_folds_partition_the_data():
    labels = np.random.default_rng(4).binomial(1, 0.2, 500)
    folds = stratified_kfold_indices(labels, n_splits=5, seed=42)
    combined = np.concatenate([fold.test for fold in folds])
    assert sorted(combined) == list(range(500))


def test_make_splits_dispatches_on_config(timed_frame):
    assert len(make_splits(timed_frame, {"type": "time", "column": "t"}, "y", 42)) == 1
    assert len(make_splits(timed_frame, {"type": "stratified_kfold", "n_splits": 5}, "y", 42)) == 5
    assert len(make_splits(timed_frame, {"type": "random", "test_size": 0.2}, "y", 42)) == 1


def test_make_splits_rejects_an_unknown_strategy(timed_frame):
    with pytest.raises(ValueError, match="Unknown split.type"):
        make_splits(timed_frame, {"type": "leave_one_out"}, "y", 42)
