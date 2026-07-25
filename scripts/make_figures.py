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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# Non-interactive backend: this runs in CI and over SSH, where no display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

from src.config import PROBLEMS, get_project_root, load_config
from src.data.adapters import get_adapter
from src.data.split import make_splits
from src.serving.registry import CheckpointRegistry
from src.training.tabular import TabularTrainer

console = Console()

FIGURE_DIR = get_project_root() / "reports" / "figures"
IMAGE_DIR = get_project_root() / "docs" / "images"
DPI = 160


def _rebuild_test_split(problem: str):
    """Reconstruct the exact test split the checkpoint was scored on.

    The split is deterministic given the config and the seed, so this reproduces
    it rather than storing a copy of the test set on disk.
    """
    config = load_config(problem)
    adapter = get_adapter(config["data"]["source"]["adapter"])
    frame = adapter.load()
    trainer = TabularTrainer(problem)
    split = make_splits(frame, config["split"], config["data"]["target"], config["seed"])[0]
    data = trainer.build_matrix(frame, split)
    trainer.finish()
    return data


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    console.print(f"    {path.relative_to(get_project_root())}")
    return path


def pr_curve(problem: str, y_true, scores: dict[str, np.ndarray]) -> Path:
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
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_pr_curve")


def roc_plot(problem: str, y_true, scores: dict[str, np.ndarray]) -> Path:
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
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_roc_curve")


def calibration(problem: str, y_true, y_score: np.ndarray) -> Path:
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
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, f"{problem}_calibration")


def confusion_at_review_budget(problem: str, y_true, y_score: np.ndarray) -> Path:
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

    data = _rebuild_test_split(problem)
    y_true = data["y_test"]
    matrix = data["X_test"]

    scores = {"LightGBM": loaded.model.predict_proba(matrix)[:, 1]}
    for baseline in load_config(problem).get("baselines", []):
        from src.models.registry import create_model

        estimator = create_model(baseline["type"], baseline.get("params"))
        estimator.fit(data["X_train"], data["y_train"])
        scores[baseline["type"]] = estimator.predict_proba(matrix)[:, 1]

    pr_curve(problem, y_true, scores)
    roc_plot(problem, y_true, scores)
    calibration(problem, y_true, scores["LightGBM"])
    confusion_at_review_budget(problem, y_true, scores["LightGBM"])
    shap_summary(problem, loaded.model, matrix)
    feature_importance(problem, loaded.model, list(matrix.columns))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--problem", choices=[*PROBLEMS, "all"], default="all")
    args = parser.parse_args()

    registry = CheckpointRegistry()
    problems = list(PROBLEMS) if args.problem == "all" else [args.problem]
    produced = sum(figures_for(problem, registry) for problem in problems)

    if produced == 0:
        console.print(
            "\n[bold red]No figures produced — no trained checkpoint exists.[/bold red]\n"
            "A blank chart in a README is worse than no chart, so nothing was drawn.\n"
            "  uv run python scripts/download_data.py --dataset all\n"
            "  uv run python scripts/train.py --model all\n"
            "See docs/PROGRESS.md."
        )
        return 1

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[green]{produced} problem(s) plotted into reports/figures/.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
