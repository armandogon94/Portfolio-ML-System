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


def test_published_sample_summary_is_derived_from_the_prediction_rows():
    frame = pd.DataFrame({"y_true": [0, 1, 0, 1]})

    assert figures._sample_summary(frame) == "2 positives in 4 rows"


def test_published_figure_titles_state_the_findings():
    assert figures.PUBLISHED_PR_TITLE == (
        "LightGBM PR point estimates exceed logistic regression on all measured sets"
    )
    assert figures.PUBLISHED_CALIBRATION_TITLE == (
        "Observed event rates rise with model scores in all three datasets"
    )


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


def test_calibration_bins_preserve_all_input_measurements():
    frame = pd.DataFrame(
        {
            "y_true": (np.arange(100) % 7 == 0).astype(int),
            "score_model": np.linspace(0.001, 0.999, 100),
        }
    )

    bins = figures._quantile_calibration_bins(
        frame["y_true"].to_numpy(),
        frame["score_model"].to_numpy(),
    )

    assert bins.count.sum() == len(frame)
    assert bins.positive_count.sum() == frame["y_true"].sum()
    assert np.average(bins.mean_predicted, weights=bins.count) == pytest.approx(
        frame["score_model"].mean()
    )
    assert bins.observed == pytest.approx(bins.positive_count / bins.count)


def test_zero_positive_wilson_interval_has_measured_upper_limit():
    lower, upper = figures._wilson_interval(
        np.asarray([0]),
        np.asarray([28481]),
    )

    assert lower.tolist() == [0.0]
    assert upper[0] == pytest.approx(1.3486e-04, rel=1e-4)


def test_wilson_interval_contains_observed_boundary_proportions():
    lower, upper = figures._wilson_interval(
        np.asarray([0, 10]),
        np.asarray([10, 10]),
    )

    assert lower[0] == 0.0
    assert upper[1] == 1.0


def test_calibration_panel_uses_log_axes_and_preserves_zero_bins():
    frame = pd.DataFrame(
        {
            "y_true": [0] * 80 + [0, 1] * 10,
            "score_model": np.linspace(0.001, 0.999, 100),
        }
    )
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
    zero_count = np.count_nonzero(bins.positive_count == 0)
    assert len(positive_line.get_xdata()) == len(bins.count) - zero_count
    assert len(zero_whiskers.get_segments()) == zero_count
    assert zero_count > 0
    plt.close(fig)
