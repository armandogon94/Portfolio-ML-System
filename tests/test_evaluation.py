"""Tests for evaluation metric modules."""

import numpy as np
import pytest

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.regression_metrics import compute_regression_metrics
from src.evaluation.timeseries_metrics import compute_timeseries_metrics


class TestClassificationMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["test_accuracy"] == 1.0
        assert metrics["test_precision"] == 1.0
        assert metrics["test_recall"] == 1.0
        assert metrics["test_f1"] == 1.0

    def test_with_probabilities(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert "test_auc_roc" in metrics
        assert metrics["test_auc_roc"] == pytest.approx(1.0)

    def test_without_probabilities(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert "test_auc_roc" not in metrics

    def test_custom_prefix(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_classification_metrics(y_true, y_pred, prefix="train")
        assert "train_accuracy" in metrics
        assert "test_accuracy" not in metrics

    def test_all_zeros_prediction(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["test_accuracy"] == 0.5
        assert metrics["test_recall"] == 0.0


class TestRegressionMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["test_mse"] == pytest.approx(0.0)
        assert metrics["test_rmse"] == pytest.approx(0.0)
        assert metrics["test_mae"] == pytest.approx(0.0)
        assert metrics["test_r2"] == pytest.approx(1.0)

    def test_known_errors(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["test_mae"] == pytest.approx(1.0)
        assert metrics["test_mse"] == pytest.approx(1.0)

    def test_custom_prefix(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        metrics = compute_regression_metrics(y_true, y_pred, prefix="val")
        assert "val_r2" in metrics


class TestTimeseriesMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics = compute_timeseries_metrics(y_true, y_pred)
        assert metrics["test_mae"] == pytest.approx(0.0)
        assert metrics["test_rmse"] == pytest.approx(0.0)
        assert metrics["test_mape"] == pytest.approx(0.0)

    def test_known_errors(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        metrics = compute_timeseries_metrics(y_true, y_pred)
        assert metrics["test_mae"] == pytest.approx(15.0)
        assert metrics["test_mape"] == pytest.approx(10.0)

    def test_custom_prefix(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        metrics = compute_timeseries_metrics(y_true, y_pred, prefix="val")
        assert "val_mae" in metrics
