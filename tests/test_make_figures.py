"""Figure data sources: cross-validation plots must use held-out predictions."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import scripts.make_figures as figures


def test_cv_figure_loader_reads_persisted_out_of_fold_predictions(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(
        {
            "row_index": [2, 0, 3, 1],
            "fold": [1, 1, 2, 2],
            "y_true": [1, 0, 0, 1],
            "score_model": [0.8, 0.1, 0.2, 0.7],
            "score_prior": [0.5, 0.5, 0.5, 0.5],
        }
    ).to_csv(reports / "churn_oof_predictions.csv", index=False)

    labels, scores = figures._load_oof_predictions("churn", reports_dir=reports)

    assert labels.tolist() == [0, 1, 1, 0]
    assert scores["LightGBM"].tolist() == [0.1, 0.7, 0.8, 0.2]
    assert scores["prior"].tolist() == [0.5, 0.5, 0.5, 0.5]


def test_all_problem_run_fails_when_any_checkpoint_is_missing(monkeypatch):
    monkeypatch.setattr(figures, "available_config_names", lambda: ["fraud", "churn"])
    monkeypatch.setattr(figures, "CheckpointRegistry", object)
    monkeypatch.setattr(
        figures,
        "figures_for",
        lambda problem, registry: int(problem == "churn"),
    )
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--problem", "all"])

    assert figures.main() == 1


@pytest.mark.parametrize(
    ("problem", "expected_counts", "expected_positives", "expected_predicted"),
    [
        (
            "fraud_ulb",
            [28481, 28481, 28481, 28480, 28481, 28480, 28481, 28480, 28481, 28481],
            [0, 1, 0, 3, 4, 0, 3, 3, 9, 469],
            [
                3.60129588232e-11,
                8.88821374096e-11,
                1.55863369795e-10,
                2.51357862813e-10,
                3.99009983444e-10,
                6.42700387867e-10,
                1.09187114224e-09,
                2.07064127505e-09,
                5.20650545593e-09,
                1.43880138262e-02,
            ],
        ),
        (
            "churn",
            [1013, 1013, 1012, 1013, 1013, 1012, 1013, 1012, 1013, 1013],
            [0, 0, 0, 0, 1, 2, 3, 33, 585, 1003],
            [
                1.34368391407e-06,
                3.67722316874e-06,
                7.31560755380e-06,
                1.43589637040e-05,
                3.02055297884e-05,
                7.39547107934e-05,
                2.74409124280e-04,
                4.75228449633e-03,
                5.41802215718e-01,
                9.98670941901e-01,
            ],
        ),
    ],
)
def test_published_calibration_bins_match_persisted_oof_measurements(
    problem, expected_counts, expected_positives, expected_predicted
):
    frame = pd.read_csv(figures.REPORTS_DIR / f"{problem}_oof_predictions.csv")

    bins = figures._quantile_calibration_bins(
        frame["y_true"].to_numpy(),
        frame["score_model"].to_numpy(),
    )

    assert bins.count.tolist() == expected_counts
    assert bins.positive_count.tolist() == expected_positives
    assert bins.mean_predicted == pytest.approx(expected_predicted, rel=1e-9)
    assert bins.observed == pytest.approx(
        np.asarray(expected_positives) / np.asarray(expected_counts)
    )


def test_zero_positive_wilson_interval_has_measured_upper_limit():
    lower, upper = figures._wilson_interval(
        np.asarray([0]),
        np.asarray([28481]),
    )

    assert lower.tolist() == [0.0]
    assert upper[0] == pytest.approx(1.3486e-04, rel=1e-4)


def test_calibration_panel_uses_log_axes_and_preserves_zero_bins():
    frame = pd.read_csv(figures.REPORTS_DIR / "fraud_ulb_oof_predictions.csv")
    bins = figures._quantile_calibration_bins(
        frame["y_true"].to_numpy(),
        frame["score_model"].to_numpy(),
    )
    fig, ax = plt.subplots()

    figures._draw_calibration_panel(ax, bins, color="#2563eb")

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    positive_line = next(line for line in ax.lines if line.get_gid() == "positive-bins")
    zero_whiskers = next(
        collection for collection in ax.collections if collection.get_gid() == "zero-bin-whiskers"
    )
    assert len(positive_line.get_xdata()) == 7
    assert len(zero_whiskers.get_segments()) == 3
    assert np.count_nonzero(bins.positive_count == 0) == 3
    plt.close(fig)
