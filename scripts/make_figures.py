#!/usr/bin/env python
"""Regenerate every figure in the README and reports/ from real checkpoints.

Figures are **never hand-exported**. Everything under `reports/figures/` and every
chart in `docs/images/` is produced by this script from a checkpoint that a
training run wrote, so a figure cannot drift from the number beside it.

Produces, per problem:
  reports/figures/<problem>_pr_curve.png            model vs each baseline
  reports/figures/<problem>_roc_curve.png
  reports/figures/<problem>_calibration.png         predicted vs observed rate
  reports/figures/<problem>_confusion_matrix.png    at the 1%-review threshold
  reports/figures/<problem>_shap_summary.png        top-20 mean |SHAP|
  reports/figures/<problem>_feature_importance.png  LightGBM gain

Requires a trained checkpoint. Without one it says so and exits non-zero rather
than drawing an empty axis; a blank chart in a README is worse than no chart.

The two README figures are rebuilt directly from persisted out-of-fold scores:
  reports/figures/precision_recall_curves.png
  reports/figures/calibration_curves.png

Usage:
    uv run python scripts/make_figures.py --published-only
    uv run python scripts/make_figures.py
    uv run python scripts/make_figures.py --problem fraud
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# Non-interactive backend: this runs in CI and over SSH, where no display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import available_config_names, get_project_root, load_config
from src.data.adapters import get_adapter
from src.data.split import make_splits
from src.features.schema import apply_category_dtypes
from src.serving.registry import CheckpointRegistry

console = Console()

REPORTS_DIR = get_project_root() / "reports"
FIGURE_DIR = get_project_root() / "reports" / "figures"
IMAGE_DIR = get_project_root() / "docs" / "images"
DPI = 160
WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class CalibrationBins:
    """Measured probability-bin summaries and binomial uncertainty."""

    mean_predicted: np.ndarray
    observed: np.ndarray
    positive_count: np.ndarray
    count: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def _rebuild_model_matrix(problem: str, loaded):
    """Rebuild feature matrices with the exact state saved in the checkpoint."""
    config = load_config(problem)
    adapter = get_adapter(config["data"]["source"]["adapter"])
    frame = adapter.load()
    features = importlib.import_module(config["features"]["module"])
    engineered, _ = features.engineer_features(frame, loaded.feature_artifacts, fit=False)
    matrix = engineered.reindex(columns=loaded.feature_columns)
    for column in matrix.columns:
        if str(matrix[column].dtype) == "object":
            matrix[column] = matrix[column].astype("category")
    matrix = apply_category_dtypes(matrix, loaded.category_dtypes)
    return frame, matrix


def _load_oof_frame(problem: str, *, reports_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate persisted held-out predictions."""
    root = reports_dir or REPORTS_DIR
    path = root / f"{problem}_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cross-validation figures require {path}. Retrain {problem!r}; "
            "the trainer writes one held-out prediction for every row."
        )
    frame = pd.read_csv(path).sort_values("row_index")
    required = {"row_index", "fold", "y_true", "score_model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required OOF columns: {sorted(missing)}")
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate OOF row_index values.")
    if not frame["y_true"].isin([0, 1]).all():
        raise ValueError(f"{path} contains labels outside 0 and 1.")
    return frame


def _load_oof_predictions(
    problem: str, *, reports_dir: Path | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load held-out predictions for honest cross-validation performance plots."""
    frame = _load_oof_frame(problem, reports_dir=reports_dir)

    model_type = load_config(problem)["model"]["type"]
    labels = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "logreg": "Logistic regression",
        "prior": "prior",
    }
    scores: dict[str, np.ndarray] = {
        labels.get(model_type, model_type): frame["score_model"].to_numpy()
    }
    for column in frame.columns:
        if column.startswith("score_") and column != "score_model":
            name = column.removeprefix("score_")
            scores[labels.get(name, name)] = frame[column].to_numpy()
    return frame["y_true"].to_numpy(), scores


def _wilson_interval(
    positive_count: np.ndarray,
    count: np.ndarray,
    *,
    z: float = WILSON_Z_95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a two-sided Wilson score interval for each binomial count."""
    positive = np.asarray(positive_count, dtype=float)
    total = np.asarray(count, dtype=float)
    if positive.shape != total.shape:
        raise ValueError("positive_count and count must have the same shape.")
    if np.any(total <= 0) or np.any(positive < 0) or np.any(positive > total):
        raise ValueError("Binomial counts must satisfy 0 <= positive_count <= count.")

    proportion = positive / total
    z_squared = z**2
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total**2))
        / denominator
    )
    return np.maximum(0.0, centre - half_width), np.minimum(1.0, centre + half_width)


def _quantile_calibration_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationBins:
    """Summarise quantile bins without dropping bins that have zero positives."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("y_score must contain finite probabilities in [0, 1].")

    edges = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.searchsorted(edges[1:-1], scores)
    count = np.bincount(bin_ids, minlength=n_bins)
    if np.any(count == 0):
        raise ValueError(
            f"Quantile binning produced {np.count_nonzero(count == 0)} empty bins; "
            "the published figure requires ten measured bins."
        )

    positive_count = np.bincount(bin_ids, weights=labels, minlength=n_bins).astype(int)
    score_sum = np.bincount(bin_ids, weights=scores, minlength=n_bins)
    mean_predicted = score_sum / count
    observed = positive_count / count
    lower, upper = _wilson_interval(positive_count, count)
    return CalibrationBins(
        mean_predicted=mean_predicted,
        observed=observed,
        positive_count=positive_count,
        count=count,
        lower=lower,
        upper=upper,
    )


def _metric_value(problem: str, metric: str, *, reports_dir: Path | None = None) -> float:
    """Read one training-written metric and fail if it is absent or ambiguous."""
    root = reports_dir or REPORTS_DIR
    path = root / f"{problem}_metrics.csv"
    frame = pd.read_csv(path)
    matches = frame.loc[frame["metric"] == metric, "value"]
    if len(matches) != 1:
        raise ValueError(f"{path} must contain exactly one {metric!r} row.")
    return float(matches.iloc[0])


def _validated_fold_metric(
    problem: str,
    score_column: str,
    metric_name: str,
    scorer,
    *,
    reports_dir: Path | None = None,
) -> float:
    """Cross-check a persisted fold metric against the OOF rows used in a figure."""
    frame = _load_oof_frame(problem, reports_dir=reports_dir)
    measured = float(
        frame.groupby("fold", sort=True)
        .apply(
            lambda fold: scorer(fold["y_true"], fold[score_column]),
            include_groups=False,
        )
        .mean()
    )
    recorded = _metric_value(problem, metric_name, reports_dir=reports_dir)
    if not np.isclose(measured, recorded, rtol=1e-12, atol=1e-15):
        raise ValueError(
            f"{problem} {metric_name} does not match its persisted OOF predictions: "
            f"metrics CSV={recorded!r}, OOF rows={measured!r}."
        )
    return recorded


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    console.print(f"    {path.relative_to(get_project_root())}")
    return path


def _published_precision_recall(*, reports_dir: Path | None = None) -> Path:
    """Build the README PR figure from the two persisted OOF score files."""
    root = reports_dir or REPORTS_DIR
    specifications = (
        ("fraud_ulb", "ULB fraud", "492 positives in 284,807 rows"),
        ("churn", "Card attrition", "1,627 positives in 10,127 rows"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1))
    colours = {"score_model": "#2563eb", "score_logreg": "#ea580c"}
    labels = {"score_model": "LightGBM", "score_logreg": "Logistic regression"}

    for ax, (problem, title, sample_text) in zip(axes, specifications, strict=True):
        frame = _load_oof_frame(problem, reports_dir=root)
        for column in ("score_model", "score_logreg"):
            precision, recall, _ = precision_recall_curve(
                frame["y_true"],
                frame[column],
            )
            pooled_ap = average_precision_score(frame["y_true"], frame[column])
            ax.plot(
                recall,
                precision,
                color=colours[column],
                linewidth=2.2,
                label=f"{labels[column]} (pooled AP {pooled_ap:.3f})",
            )

        base_rate = float(frame["y_true"].mean())
        ax.axhline(
            base_rate,
            color="#6b7280",
            linestyle="--",
            linewidth=1.3,
            label=f"positive rate {base_rate:.4f}",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}\n{sample_text}", fontsize=11)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Precision-recall performance on held-out rows", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.82, wspace=0.22)
    return _save(fig, "precision_recall_curves")


def _log_tick_label(value: float, _position: int) -> str:
    """Format exact powers of ten as compact labels such as 1e-9."""
    if value <= 0:
        return ""
    exponent = int(round(np.log10(value)))
    if not np.isclose(value, 10.0**exponent):
        return ""
    return "1" if exponent == 0 else f"1e{exponent}"


def _calibration_y_floor(bins: CalibrationBins) -> float:
    """Place the floor just below the smallest measured positive interval limit."""
    finite_positive = np.concatenate(
        (
            bins.observed[bins.observed > 0],
            bins.lower[bins.lower > 0],
            bins.upper[bins.upper > 0],
        )
    )
    if finite_positive.size == 0:
        raise ValueError("Calibration bins contain no finite positive interval limit.")
    return float(finite_positive.min() / 1.6)


def _draw_calibration_panel(ax, bins: CalibrationBins, *, color: str) -> None:
    """Draw all quantile bins, including zero-event bins, on log axes."""
    positive = bins.positive_count > 0
    zero = ~positive
    y_floor = _calibration_y_floor(bins)

    interval = ax.errorbar(
        bins.mean_predicted[positive],
        bins.observed[positive],
        yerr=np.vstack(
            (
                bins.observed[positive] - bins.lower[positive],
                bins.upper[positive] - bins.observed[positive],
            )
        ),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.4,
        capsize=4,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3,
    )
    interval.lines[0].set_gid("positive-bins")

    if np.any(zero):
        zero_x = bins.mean_predicted[zero]
        zero_upper = bins.upper[zero]
        whiskers = ax.vlines(
            zero_x,
            y_floor,
            zero_upper,
            color="#7c3aed",
            linewidth=1.7,
            zorder=2,
        )
        whiskers.set_gid("zero-bin-whiskers")
        cap_left = zero_x / 1.12
        cap_right = zero_x * 1.12
        ax.hlines(
            zero_upper,
            cap_left,
            cap_right,
            color="#7c3aed",
            linewidth=1.7,
            zorder=2,
        )
        ax.hlines(
            np.full(zero_x.shape, y_floor),
            cap_left,
            cap_right,
            color="#7c3aed",
            linewidth=1.7,
            zorder=2,
        )

    x_lower = float(bins.mean_predicted.min() / 1.7)
    x_upper = 1.25 if bins.mean_predicted.max() > 0.8 else float(bins.mean_predicted.max() * 1.35)
    y_upper = 1.15 if bins.upper.max() > 0.8 else float(bins.upper.max() * 1.4)
    diagonal_lower = max(x_lower, y_floor)
    diagonal_upper = min(x_upper, y_upper, 1.0)
    diagonal = ax.plot(
        [diagonal_lower, diagonal_upper],
        [diagonal_lower, diagonal_upper],
        color="#4b5563",
        linestyle="--",
        linewidth=1.3,
        zorder=1,
    )[0]
    diagonal.set_gid("perfect-calibration")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_floor, y_upper)
    formatter = FuncFormatter(_log_tick_label)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=9))
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(which="major", alpha=0.22)
    ax.grid(which="minor", alpha=0.08)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency (share that are positive)")


def _published_calibration(*, reports_dir: Path | None = None) -> Path:
    """Build the README calibration figure with Wilson intervals in every bin."""
    root = reports_dir or REPORTS_DIR
    specifications = (
        ("fraud_ulb", "ULB fraud", "#2563eb"),
        ("churn", "Card attrition", "#ea580c"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.1))

    for ax, (problem, title, color) in zip(axes, specifications, strict=True):
        frame = _load_oof_frame(problem, reports_dir=root)
        bins = _quantile_calibration_bins(
            frame["y_true"].to_numpy(),
            frame["score_model"].to_numpy(),
        )
        _draw_calibration_panel(ax, bins, color=color)
        brier = _validated_fold_metric(
            problem,
            "score_model",
            "cv_brier_mean",
            brier_score_loss,
            reports_dir=root,
        )
        ax.set_title(f"{title}\n10 equal-count score bins", fontsize=11)
        ax.legend(
            [Line2D([], [], linestyle="none")],
            [f"Brier score {brier:.5f}"],
            loc="upper left",
            handlelength=0,
            handletextpad=0,
            framealpha=0.95,
            fontsize=9,
        )

    shared_legend = (
        Line2D(
            [],
            [],
            color="#2563eb",
            marker="o",
            linestyle="none",
            markersize=6,
            label="bin with observed positives",
        ),
        Line2D(
            [],
            [],
            color="#7c3aed",
            marker="_",
            linestyle="-",
            linewidth=1.7,
            markersize=9,
            label="bin with none observed (upper bound only)",
        ),
        Line2D(
            [],
            [],
            color="#4b5563",
            linestyle="--",
            linewidth=1.3,
            label="perfect calibration",
        ),
    )
    fig.legend(
        handles=shared_legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle("Calibration across ten probability quantiles", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.19, top=0.82, wspace=0.24)
    return _save(fig, "calibration_curves")


def pr_curve(
    problem: str, y_true, scores: dict[str, np.ndarray], *, evidence_note: str = ""
) -> Path:
    """Precision-recall curve. THE plot for an imbalanced problem.

    The horizontal line is the base rate, which is what a random ranker achieves.
    Without it a PR curve is unreadable: 0.30 average precision is excellent at
    3.5% positives and embarrassing at 40%.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})", linewidth=2)

    base_rate = float(np.mean(y_true))
    ax.axhline(
        base_rate,
        linestyle="--",
        color="grey",
        linewidth=1,
        label=f"base rate = {base_rate:.4f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{problem}: precision-recall")
    if evidence_note:
        fig.text(0.5, 0.01, evidence_note, ha="center", fontsize=8, color="#555555")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_pr_curve")


def roc_plot(
    problem: str, y_true, scores: dict[str, np.ndarray], *, evidence_note: str = ""
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(
            fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_true, y_score):.3f})", linewidth=2
        )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{problem}: ROC (secondary; see the PR curve first)")
    if evidence_note:
        fig.text(0.5, 0.01, evidence_note, ha="center", fontsize=8, color="#555555")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_roc_curve")


def calibration(problem: str, y_true, y_score: np.ndarray, *, evidence_note: str = "") -> Path:
    """Are the probabilities honest, not just well-ordered?

    AUC only cares about ranking. A model can rank perfectly and still say "90%"
    about things that happen 30% of the time, which makes every downstream
    threshold meaningless.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    observed, predicted = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    ax.plot(predicted, observed, "o-", linewidth=2, label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{problem}: calibration (10 quantile bins)")
    if evidence_note:
        fig.text(0.5, 0.01, evidence_note, ha="center", fontsize=8, color="#555555")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_calibration")


def confusion_at_review_budget(
    problem: str, y_true, y_score: np.ndarray, *, evidence_note: str = ""
) -> Path:
    """Confusion matrix at the top-1% operating point, not at 0.5.

    A 0.5 threshold on a 3.5%-positive problem predicts almost nothing positive
    and produces a confusion matrix that says only "the classes are imbalanced".
    The top-1% cut is the operating point a review team actually works at.
    """
    n_review = max(1, int(round(len(y_true) * 0.01)))
    threshold = float(np.sort(y_score)[::-1][n_review - 1])
    y_pred = (y_score >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=["negative", "positive"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{problem}: top 1% reviewed\n(threshold = {threshold:.4f})", fontsize=10)
    if evidence_note:
        fig.text(0.5, 0.01, evidence_note, ha="center", fontsize=8, color="#555555")
    return _save(fig, f"{problem}_confusion_matrix")


def shap_summary(problem: str, model, matrix) -> Path | None:
    """Mean |SHAP| over a sample of the test split."""
    import shap

    sample = matrix.sample(n=min(2000, len(matrix)), random_state=42)
    try:
        values = shap.TreeExplainer(model).shap_values(sample)
    except Exception as exc:  # noqa: BLE001 - SHAP raises many library-specific types
        console.print(f"    [yellow]SHAP unavailable for {problem}: {exc}[/yellow]")
        return None

    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh(range(len(order)), mean_abs[order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([sample.columns[i] for i in order], fontsize=8)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"{problem}: top 20 features by SHAP")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, f"{problem}_shap_summary")


def feature_importance(problem: str, model, columns) -> Path | None:
    """LightGBM gain. Shown alongside SHAP because they disagree, informatively.

    Gain is a training-time statistic and over-credits high-cardinality splits;
    SHAP attributes actual predictions. When they disagree the gap is worth a
    paragraph in reports/RESULTS.md.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    order = np.argsort(importances)[::-1][:20][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh(range(len(order)), np.asarray(importances)[order], color="#6b8e23")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([columns[i] for i in order], fontsize=8)
    ax.set_xlabel("LightGBM gain")
    ax.set_title(f"{problem}: top 20 features by gain")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, f"{problem}_feature_importance")


def figures_for(problem: str, registry: CheckpointRegistry) -> int:
    console.rule(f"[bold cyan]{problem}")
    try:
        loaded = registry.load(problem)
    except FileNotFoundError as exc:
        console.print(f"[yellow]skipped: {exc}[/yellow]")
        return 0

    config = load_config(problem)
    frame, full_matrix = _rebuild_model_matrix(problem, loaded)
    model_label = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "logreg": "Logistic regression",
        "prior": "prior",
    }.get(config["model"]["type"], config["model"]["type"])

    if config["split"]["type"] == "stratified_kfold":
        y_true, scores = _load_oof_predictions(problem)
        evidence_note = (
            "Out-of-fold predictions: every row was scored by a fold that did not train on it."
        )
    else:
        split = make_splits(frame, config["split"], config["data"]["target"], config["seed"])[0]
        y_true = frame.iloc[split.test][config["data"]["target"]].to_numpy()
        matrix = full_matrix.iloc[split.test]
        scores = {model_label: loaded.model.predict_proba(matrix)[:, 1]}
        for baseline in config.get("baselines", []):
            from src.models.registry import create_model

            estimator = create_model(baseline["type"], baseline.get("params"))
            estimator.fit(
                full_matrix.iloc[split.train],
                frame.iloc[split.train][config["data"]["target"]].to_numpy(),
            )
            scores[baseline["type"]] = estimator.predict_proba(matrix)[:, 1]
        evidence_note = "Held-out temporal test partition; no training row contributes a score."

    pr_curve(problem, y_true, scores, evidence_note=evidence_note)
    roc_plot(problem, y_true, scores, evidence_note=evidence_note)
    calibration(problem, y_true, scores[model_label], evidence_note=evidence_note)
    confusion_at_review_budget(problem, y_true, scores[model_label], evidence_note=evidence_note)
    shap_summary(problem, loaded.model, full_matrix)
    feature_importance(problem, loaded.model, list(full_matrix.columns))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    config_names = available_config_names()
    parser.add_argument("--problem", choices=[*config_names, "all"], default="all")
    parser.add_argument(
        "--published-only",
        action="store_true",
        help="Build the two README figures from persisted out-of-fold predictions.",
    )
    args = parser.parse_args()

    if args.published_only:
        _published_precision_recall()
        _published_calibration()
        console.print("\n[green]2 published figures rebuilt from OOF predictions.[/green]")
        return 0

    registry = CheckpointRegistry()
    problems = list(config_names) if args.problem == "all" else [args.problem]
    produced = sum(figures_for(problem, registry) for problem in problems)
    train_target = "all" if args.problem == "all" else args.problem

    if produced == 0:
        console.print(
            "\n[bold red]No figures produced: no trained checkpoint exists.[/bold red]\n"
            "A blank chart in a README is worse than no chart, so nothing was drawn.\n"
            "  uv run python scripts/download_data.py --dataset all\n"
            f"  uv run python scripts/train.py --model {train_target}\n"
            "See docs/PROGRESS.md."
        )
        return 1

    if produced != len(problems):
        console.print(
            f"\n[bold red]Incomplete figure set: {produced}/{len(problems)} "
            "requested problems had checkpoints.[/bold red]\n"
            "Train every requested problem, then regenerate the figures:\n"
            f"  uv run python scripts/train.py --model {train_target}\n"
            f"  uv run python scripts/make_figures.py --problem {args.problem}"
        )
        return 1

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[green]{produced} problem(s) plotted into reports/figures/.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
