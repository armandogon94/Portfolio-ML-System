"""Focused tests for the regenerable split-geometry report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.describe_split import _overlap, _partition_record
from src.data.split import Split


def test_overlap_reports_only_split_key_values_that_cross_a_boundary():
    january = pd.Timestamp("2020-01-01")
    frame = pd.DataFrame(
        {
            "issue_d": [
                january,
                january,
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-03-01"),
                pd.Timestamp("2020-03-01"),
            ]
        }
    )
    disjoint = Split(
        train=np.array([0, 1]),
        val=np.array([2, 3]),
        test=np.array([4, 5]),
    )
    straddled = Split(
        train=np.array([0]),
        val=np.array([1, 2, 3]),
        test=np.array([4, 5]),
    )

    assert _overlap(frame, [disjoint], "issue_d") == []
    assert _overlap(frame, [straddled], "issue_d") == [str(january)]


def test_partition_record_reports_counts_rate_and_split_key_range():
    frame = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "is_default": [0, 0, 1, 1],
        }
    )

    record = _partition_record(
        "test",
        frame,
        np.array([0, 2, 3]),
        "issue_d",
        "is_default",
    )

    assert record == {
        "partition": "test",
        "n": 3,
        "positive_rate": pytest.approx(2 / 3),
        "n_positive": 2,
        "min": "2020-01-01 00:00:00",
        "max": "2020-04-01 00:00:00",
    }


def test_partition_record_for_empty_indices_reports_only_identity_and_size():
    frame = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2020-01-01"]),
            "is_default": [0],
        }
    )

    assert _partition_record(
        "empty",
        frame,
        np.array([], dtype=int),
        "issue_d",
        "is_default",
    ) == {"partition": "empty", "n": 0}
