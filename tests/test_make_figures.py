"""Figure data sources: cross-validation plots must use held-out predictions."""

from __future__ import annotations

import sys

import pandas as pd

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
