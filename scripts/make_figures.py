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
than drawing an empty axis — a blank chart in a README is worse than no chart.

Usage:
    uv run python scripts/make_figures.py
    uv run python scripts/make_figures.py --problem fraud
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# Non-interactive backend: this runs in CI and over SSH, where no display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
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

FIGURE_DIR = get_project_root() / "reports" / "figures"
IMAGE_DIR = get_project_root() / "docs" / "images"
DPI = 160


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


def _load_oof_predictions(
    problem: str, *, reports_dir: Path | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load held-out predictions for honest cross-validation performance plots."""
    root = reports_dir or get_project_root() / "reports"
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


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    console.print(f"    {path.relative_to(get_project_root())}")
    return path


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
    ax.set_title(f"{problem}: ROC (secondary — see the PR curve first)")
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
    args = parser.parse_args()

    registry = CheckpointRegistry()
    problems = list(config_names) if args.problem == "all" else [args.problem]
    produced = sum(figures_for(problem, registry) for problem in problems)
    train_target = "all" if args.problem == "all" else args.problem

    if produced == 0:
        console.print(
            "\n[bold red]No figures produced — no trained checkpoint exists.[/bold red]\n"
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
