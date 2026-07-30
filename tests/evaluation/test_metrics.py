"""Metrics are checked against sklearn references and hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.classification_metrics import (
    aggregate_folds,
    compute_classification_metrics,
    precision_at_k,
    recall_at_fpr,
    temporal_block_spread,
)


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(7)
    labels = rng.binomial(1, 0.035, 2000)
    # Scores correlated with the label but far from perfect.
    scores = np.clip(labels * 0.4 + rng.normal(0.3, 0.2, 2000), 0, 1)
    return labels, scores


def test_pr_auc_and_roc_auc_match_sklearn(imbalanced):
    labels, scores = imbalanced
    metrics = compute_classification_metrics(labels, scores)
    assert metrics["test_pr_auc"] == pytest.approx(average_precision_score(labels, scores))
    assert metrics["test_roc_auc"] == pytest.approx(roc_auc_score(labels, scores))


def test_precision_at_k_is_hand_computable():
    """Top 2 of 10 by score contain one positive -> precision 0.5."""
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.05])
    assert precision_at_k(labels, scores, k=0.2) == pytest.approx(0.5)


def test_precision_at_k_returns_nan_when_the_slice_would_be_empty():
    labels = np.array([0, 1])
    assert np.isnan(precision_at_k(labels, np.array([0.1, 0.9]), k=0.001))


def test_recall_at_fpr_respects_the_budget():
    """A perfect ranking catches everything within any positive FPR budget."""
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.9, 1.0, 10)])
    assert recall_at_fpr(labels, scores, max_fpr=0.01) == pytest.approx(1.0)


def test_recall_at_fpr_is_nan_on_single_class_labels():
    assert np.isnan(recall_at_fpr(np.zeros(10), np.random.rand(10)))


def test_roc_auc_is_omitted_rather_than_faked_on_single_class_labels():
    """A placeholder here would end up in a results table."""
    metrics = compute_classification_metrics(np.zeros(20), np.random.rand(20))
    assert "test_roc_auc" not in metrics
    assert "test_pr_auc" in metrics


def test_positive_rate_and_n_are_reported(imbalanced):
    labels, scores = imbalanced
    metrics = compute_classification_metrics(labels, scores)
    assert metrics["test_positive_rate"] == pytest.approx(labels.mean())
    assert metrics["test_n"] == 2000


def test_prefix_is_applied(imbalanced):
    labels, scores = imbalanced
    metrics = compute_classification_metrics(labels, scores, prefix="baseline_logreg")
    assert "baseline_logreg_pr_auc" in metrics


def test_aggregate_folds_reports_mean_and_std():
    """On a 10k-row dataset, one fold's number is noise. Both are required."""
    folds = [
        {"test_roc_auc": 0.90, "test_pr_auc": 0.5},
        {"test_roc_auc": 0.94, "test_pr_auc": 0.6},
        {"test_roc_auc": 0.92, "test_pr_auc": 0.55},
    ]
    out = aggregate_folds(folds)
    assert out["cv_roc_auc_mean"] == pytest.approx(0.92)
    assert out["cv_roc_auc_std"] == pytest.approx(np.std([0.90, 0.94, 0.92], ddof=1))
    assert out["cv_n_folds"] == 3.0


def test_aggregate_folds_only_keeps_metrics_present_in_every_fold():
    out = aggregate_folds([{"test_a": 1.0, "test_b": 2.0}, {"test_a": 3.0}])
    assert "cv_a_mean" in out
    assert "cv_b_mean" not in out


def test_aggregate_folds_handles_a_single_fold():
    out = aggregate_folds([{"test_a": 1.0}])
    assert out["cv_a_mean"] == 1.0
    assert out["cv_a_std"] == 0.0


def test_aggregate_folds_on_no_folds_is_empty():
    assert aggregate_folds([]) == {}


def test_temporal_block_spread_reports_variation_without_crossing_time_ties():
    labels = np.array([0, 1] * 8)
    scores = np.array(
        [
            0.1,
            0.9,
            0.2,
            0.8,
            0.3,
            0.7,
            0.4,
            0.6,
            0.6,
            0.4,
            0.7,
            0.3,
            0.8,
            0.2,
            0.9,
            0.1,
        ]
    )
    times = np.repeat(np.arange(8), 2)

    spread = temporal_block_spread(labels, scores, times, n_blocks=4)

    block_pr = [
        average_precision_score(labels[start : start + 4], scores[start : start + 4])
        for start in range(0, 16, 4)
    ]
    assert spread["test_pr_auc_temporal_block_std"] == pytest.approx(np.std(block_pr, ddof=1))
    assert spread["test_n_temporal_blocks"] == 4.0
