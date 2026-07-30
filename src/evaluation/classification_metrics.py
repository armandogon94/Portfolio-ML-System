"""Classification metrics for heavily imbalanced binary problems.

**PR-AUC (average precision) is the primary metric here, not ROC-AUC.** ROC-AUC
measures pairwise ranking and does not express positive predictive value at a
rare base rate. Average precision responds to precision across recall levels,
which is closer to the ranking behavior a fraud-review queue needs.

Two operational metrics matter more than either AUC to a fintech reviewer:

``precision_at_k``
    Of the top k% of transactions by score, the ones a human review team would
    actually look at, what fraction are truly fraud? The queue's hit rate.

``recall_at_fpr``
    At a false-positive rate the business will tolerate (1%), what fraction of
    fraud do we catch? The loss-prevention number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, *, k: float = 0.01) -> float:
    """Precision within the top ``k`` fraction of rows ranked by score.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted positive-class scores.
        k: Fraction of the population reviewed, e.g. ``0.01`` for the top 1%.

    Returns:
        Precision in the reviewed slice, or ``nan`` if the slice would be empty.
    """
    n = len(y_true)
    n_review = int(round(n * k))
    if n_review < 1:
        return float("nan")
    top = np.argsort(y_score)[::-1][:n_review]
    return float(np.asarray(y_true)[top].mean())


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, *, max_fpr: float = 0.01) -> float:
    """Recall at the highest threshold whose false-positive rate is <= ``max_fpr``.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted positive-class scores.
        max_fpr: The false-positive budget, e.g. ``0.01``.

    Returns:
        True-positive rate at that operating point, or ``nan`` when the labels
        are single-class.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return float("nan")
    allowed = fpr <= max_fpr
    if not allowed.any():
        return float("nan")
    return float(tpr[allowed].max())


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    prefix: str = "test",
    calibrated: bool = True,
) -> dict[str, float]:
    """Compute the full reported metric set for one split.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted positive-class probabilities, or an uncalibrated
            ranking score when ``calibrated=False``.
        threshold: Decision threshold for the point metrics. For probability
            models this defaults to 0.5; anomaly scorers must pass a threshold
            fitted on training data.
        prefix: Metric-name prefix, e.g. ``"test"`` or ``"val"``.
        calibrated: Whether ``y_score`` is a genuine probability. Brier score is
            omitted for uncalibrated ranking scores rather than manufacturing a
            probability with a test-set-dependent transform.

    Returns:
        ``{metric_name: value}``. Metrics that cannot be computed on single-class
        labels are omitted rather than filled with a placeholder.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    metrics: dict[str, float] = {
        f"{prefix}_pr_auc": float(average_precision_score(y_true, y_score)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_precision_at_1pct": precision_at_k(y_true, y_score, k=0.01),
        f"{prefix}_recall_at_1pct_fpr": recall_at_fpr(y_true, y_score, max_fpr=0.01),
        f"{prefix}_positive_rate": float(y_true.mean()),
        f"{prefix}_score_std": float(y_score.std()),
        f"{prefix}_n": float(len(y_true)),
    }
    if calibrated:
        metrics[f"{prefix}_brier"] = float(brier_score_loss(y_true, y_score))

    # roc_auc_score raises on single-class y_true. That happens on tiny fixtures
    # and it is not an error worth aborting a run for.
    if len(np.unique(y_true)) > 1:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, y_score))

    return metrics


def aggregate_folds(per_fold: list[dict[str, float]], *, prefix: str = "cv") -> dict[str, float]:
    """Collapse per-fold metrics into mean and standard deviation.

    Required for the churn problem: on 10,127 rows a single fold's number is
    noise. Reporting mean AND standard deviation is the house rule.

    Args:
        per_fold: One metrics dict per fold.
        prefix: Prefix for the aggregated names, e.g. ``"cv"`` gives
            ``cv_roc_auc_mean`` and ``cv_roc_auc_std``.

    Returns:
        ``{f"{prefix}_{metric}_mean": ..., f"{prefix}_{metric}_std": ...}`` for
        every metric present in *every* fold.
    """
    if not per_fold:
        return {}

    shared = set(per_fold[0])
    for fold in per_fold[1:]:
        shared &= set(fold)

    out: dict[str, float] = {f"{prefix}_n_folds": float(len(per_fold))}
    for name in sorted(shared):
        values = np.array([fold[name] for fold in per_fold], dtype=float)
        # Strip the per-split prefix ("test_pr_auc" -> "pr_auc") so the
        # aggregated name reads cv_pr_auc_mean, not cv_test_pr_auc_mean.
        bare = name.split("_", 1)[1] if "_" in name else name
        out[f"{prefix}_{bare}_mean"] = float(np.nanmean(values))
        out[f"{prefix}_{bare}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def temporal_block_spread(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_values: np.ndarray,
    *,
    n_blocks: int = 5,
    prefix: str = "test",
    comparison_score: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure metric variation across tie-safe chronological test blocks.

    The full held-out metric remains the point estimate. The returned standard
    deviation describes temporal heterogeneity across adjacent test blocks; it
    is not a confidence interval.
    """
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    times = np.asarray(time_values)
    if labels.ndim != 1 or scores.shape != labels.shape or times.shape != labels.shape:
        raise ValueError("labels, scores, and time_values must be equal-length vectors.")
    if n_blocks < 2 or len(labels) < n_blocks:
        raise ValueError("Temporal spread requires at least two nonempty blocks.")
    comparison = None if comparison_score is None else np.asarray(comparison_score, dtype=float)
    if comparison is not None and comparison.shape != labels.shape:
        raise ValueError("comparison_score must match the label vector.")
    missing_time = np.asarray(
        [value is None or (isinstance(value, float) and np.isnan(value)) for value in times]
    )
    if missing_time.any():
        raise ValueError("Temporal spread cannot place a row with a missing time value.")

    order = np.argsort(times, kind="stable")
    ordered_times = times[order]
    boundaries = [0]
    for target in [int(round(len(order) * index / n_blocks)) for index in range(1, n_blocks)]:
        cut = target
        while cut < len(order) and ordered_times[cut - 1] == ordered_times[cut]:
            cut += 1
        if cut <= boundaries[-1] or cut >= len(order):
            raise ValueError("Time ties prevent the requested number of nonempty temporal blocks.")
        boundaries.append(cut)
    boundaries.append(len(order))

    block_metrics = []
    comparison_metrics = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        indices = order[start:end]
        if len(np.unique(labels[indices])) < 2:
            raise ValueError(
                "A temporal test block contains one class; use fewer blocks before "
                "reporting a PR-AUC or ROC-AUC spread."
            )
        block_metrics.append(
            compute_classification_metrics(labels[indices], scores[indices], prefix="block")
        )
        if comparison is not None:
            comparison_metrics.append(
                compute_classification_metrics(
                    labels[indices],
                    comparison[indices],
                    prefix="comparison",
                )
            )

    out = {f"{prefix}_n_temporal_blocks": float(len(block_metrics))}
    for metric in (
        "pr_auc",
        "roc_auc",
        "precision_at_1pct",
        "recall_at_1pct_fpr",
        "brier",
    ):
        values = np.array([block[f"block_{metric}"] for block in block_metrics], dtype=float)
        out[f"{prefix}_{metric}_temporal_block_std"] = float(np.std(values, ddof=1))
    if comparison_metrics:
        deltas = np.array(
            [
                model["block_pr_auc"] - baseline["comparison_pr_auc"]
                for model, baseline in zip(block_metrics, comparison_metrics, strict=True)
            ]
        )
        out[f"{prefix}_pr_auc_delta_temporal_block_std"] = float(np.std(deltas, ddof=1))
    return out
