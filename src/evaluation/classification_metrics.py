"""Classification metrics for heavily imbalanced binary problems.

**PR-AUC (average precision) is the primary metric here, not ROC-AUC.** At 3.5%
positives (IEEE-CIS) or 0.17% (ULB), ROC-AUC is dominated by the enormous
true-negative mass and a weak model still scores 0.8+. Average precision moves
when the top of the ranking changes, which is the only part a fraud-review queue
ever sees.

Two operational metrics matter more than either AUC to a fintech reviewer:

``precision_at_k``
    Of the top k% of transactions by score — the ones a human review team would
    actually look at — what fraction are truly fraud? The queue's hit rate.

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
) -> dict[str, float]:
    """Compute the full reported metric set for one split.

    Args:
        y_true: Binary ground truth.
        y_score: Predicted positive-class probabilities.
        threshold: Decision threshold for the point metrics. 0.5 is reported for
            comparability only — the operating point that matters is expressed by
            ``precision_at_1pct`` / ``recall_at_1pct_fpr``.
        prefix: Metric-name prefix, e.g. ``"test"`` or ``"val"``.

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
        f"{prefix}_brier": float(brier_score_loss(y_true, y_score)),
        f"{prefix}_precision_at_1pct": precision_at_k(y_true, y_score, k=0.01),
        f"{prefix}_recall_at_1pct_fpr": recall_at_fpr(y_true, y_score, max_fpr=0.01),
        f"{prefix}_positive_rate": float(y_true.mean()),
        f"{prefix}_n": float(len(y_true)),
    }

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
