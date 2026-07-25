"""Figure data sources: cross-validation plots must use held-out predictions."""

from __future__ import annotations

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
