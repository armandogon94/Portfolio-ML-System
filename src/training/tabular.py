"""One config-driven trainer for all three problems.

This file replaces the eight ``src/training/train_<problem>.py`` modules that
preceded it. They were 80% identical and diverged in the 20% that mattered: two of
them applied a different split, one silently skipped early stopping. A single
implementation driven by ``configs/<problem>.yaml`` makes "config-driven training"
a fact about the repository rather than a claim in its README.

What the config controls end to end: the dataset and adapter, the feature module,
the split strategy and key, the model and its hyperparameters, the baselines, the
reported metrics, the seed, and the sanity band the result must fall inside.

Guardrails that are code, not documentation:

* ``--sample`` refuses to write a checkpoint or a metrics CSV. Fixture numbers can
  never become published numbers by accident.
* Feature-fitting state (frequency maps, group means) is fitted on train only and
  reused verbatim for validation, test and serving.
* Every denylisted column is dropped before the model sees the frame, and the drop
  is asserted rather than assumed.
* The measured metric is compared against ``expected.roc_auc_{min,max}`` and a
  breach is logged as a **suspected leak**, loudly, in the metadata.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_project_root
from src.data.adapters import get_adapter
from src.data.split import Split, make_splits
from src.evaluation.classification_metrics import (
    aggregate_folds,
    compute_classification_metrics,
)
from src.features.schema import apply_category_dtypes, capture_category_dtypes
from src.models.registry import create_model
from src.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TabularTrainer(BaseTrainer):
    """Train, evaluate and checkpoint one problem, entirely from its config.

    Args:
        config_name: A problem name resolving to ``configs/<name>.yaml``.
        use_wandb: Enable W&B logging when ``WANDB_API_KEY`` is set.
        sample: Train on the committed CI fixture instead of the real dataset.
            **Forces checkpointing off** — see :meth:`save_artifacts`.
    """

    def __init__(self, config_name: str, *, use_wandb: bool = False, sample: bool = False):
        super().__init__(config_name, use_wandb=use_wandb)
        self.sample = sample
        self.feature_columns: list[str] = []
        self.feature_artifacts: dict[str, Any] = {}
        #: Exact training category sets, replayed at serving time. See
        #: src/features/schema.py for why permuted codes are the real danger.
        self.category_dtypes: dict[str, list] = {}
        self.baseline_metrics: dict[str, dict[str, float]] = {}
        self._fold_metrics: list[dict[str, float]] = []

    # ── pipeline steps ───────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        """Load the canonical frame through the adapter named in the config."""
        source = self.config["data"]["source"]
        adapter = get_adapter(source["adapter"])
        if self.sample:
            logger.warning(
                "SAMPLE MODE: reading %s. These are synthetic CI fixtures. "
                "No checkpoint and no metrics CSV will be written.",
                self.config["data"].get("sample_path"),
            )
            return adapter.load(sample=True)
        return adapter.load()

    def build_matrix(self, frame: pd.DataFrame, split: Split) -> dict[str, Any]:
        """Engineer features and assemble train/val/test matrices for one split.

        Feature state is fitted on the training rows only. Doing it on the whole
        frame leaks the test distribution into the encodings and is worth roughly
        a point of AUC that evaporates in production.
        """
        features = importlib.import_module(self.config["features"]["module"])
        target = self.config["data"]["target"]
        denylist = list(self.config["data"].get("denylist", [])) or [target]

        train_raw = frame.iloc[split.train]
        engineered_train, self.feature_artifacts = features.engineer_features(train_raw, fit=True)
        self.feature_columns = features.get_feature_columns(engineered_train, denylist)

        self._assert_no_denylisted(denylist)

        def prepare(indices: np.ndarray) -> tuple[pd.DataFrame, np.ndarray] | None:
            if len(indices) == 0:
                return None
            engineered, _ = features.engineer_features(
                frame.iloc[indices], self.feature_artifacts, fit=False
            )
            matrix = self._align(engineered)
            return matrix, frame.iloc[indices][target].to_numpy()

        train_matrix = self._align(engineered_train, capture=True)
        data: dict[str, Any] = {
            "X_train": train_matrix,
            "y_train": train_raw[target].to_numpy(),
        }
        for name, indices in (("val", split.val), ("test", split.test)):
            prepared = prepare(indices)
            if prepared is not None:
                data[f"X_{name}"], data[f"y_{name}"] = prepared
        return data

    def train(self, data: dict[str, Any]) -> Any:
        """Fit the configured model, with early stopping when a val fold exists."""
        model_config = self.config["model"]
        params = dict(model_config.get("params", {}))
        rounds = params.pop("early_stopping_rounds", None)
        model = create_model(model_config["type"], params, seed=self.seed)

        fit_kwargs: dict[str, Any] = {}
        if rounds and "X_val" in data and model_config["type"] == "lightgbm":
            import lightgbm as lgb

            fit_kwargs["eval_set"] = [(data["X_val"], data["y_val"])]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        model.fit(data["X_train"], data["y_train"], **fit_kwargs)
        return model

    def evaluate(self, model: Any, data: dict[str, Any]) -> dict[str, float]:
        """Score the model on the test split and on every configured baseline."""
        metrics = compute_classification_metrics(
            data["y_test"], _positive_scores(model, data["X_test"]), prefix="test"
        )

        for baseline in self.config.get("baselines", []):
            name = baseline["type"]
            estimator = create_model(name, baseline.get("params"), seed=self.seed)
            estimator.fit(data["X_train"], data["y_train"])
            scored = compute_classification_metrics(
                data["y_test"],
                _positive_scores(estimator, data["X_test"]),
                prefix=f"baseline_{name}",
            )
            self.baseline_metrics[name] = scored
            metrics.update(scored)

        # Delta against the strongest baseline PR-AUC. The absolute number means
        # nothing without this; a 0.30 PR-AUC is excellent at 3.5% positives and
        # embarrassing at 40%.
        baseline_pr = [
            scored[f"baseline_{name}_pr_auc"]
            for name, scored in self.baseline_metrics.items()
            if f"baseline_{name}_pr_auc" in scored
        ]
        if baseline_pr:
            best = max(baseline_pr)
            metrics["test_pr_auc_baseline"] = best
            metrics["test_pr_auc_delta"] = metrics["test_pr_auc"] - best

        return metrics

    # ── orchestration ────────────────────────────────────────────────────────

    def run(self) -> dict[str, float]:
        """Execute the whole pipeline and return the metrics that were measured."""
        started = time.time()
        logger.info("=" * 68)
        logger.info("Training %s (sample=%s, seed=%d)", self.problem, self.sample, self.seed)
        logger.info("=" * 68)

        frame = self.load_data()
        splits = make_splits(frame, self.config["split"], self.config["data"]["target"], self.seed)
        logger.info("%d split(s) from strategy %r", len(splits), self.config["split"]["type"])

        model = None
        metrics: dict[str, float] = {}
        for index, split in enumerate(splits):
            logger.info("fold %d/%d: %s", index + 1, len(splits), split.describe())
            data = self.build_matrix(frame, split)
            model = self.train(data)
            fold_metrics = self.evaluate(model, data)
            self._fold_metrics.append(fold_metrics)
            metrics = fold_metrics

        if len(splits) > 1:
            # Cross-validated problems report mean +/- std, never one fold.
            metrics = {**metrics, **aggregate_folds(self._fold_metrics, prefix="cv")}

        metrics["n_features"] = float(len(self.feature_columns))
        metrics["training_time_seconds"] = round(time.time() - started, 1)
        self.metrics = metrics
        self.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        leak_warning = self._check_expected_band(metrics)
        self.save_artifacts(model, metrics, leak_warning=leak_warning)
        self.finish()
        return metrics

    def save_artifacts(
        self, model: Any, metrics: dict[str, float], *, leak_warning: str | None
    ) -> Path | None:
        """Write the checkpoint, metadata and metrics CSV. No-op in sample mode.

        Returns:
            The checkpoint directory, or ``None`` when nothing was written.
        """
        if self.sample:
            logger.warning(
                "SAMPLE MODE: skipping checkpoint and reports/. Fixture-derived "
                "numbers must never reach a published table."
            )
            return None

        import joblib

        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, checkpoint_dir / "model.joblib")
        joblib.dump(self.feature_bundle(), checkpoint_dir / "features.joblib")

        metadata = {
            "problem": self.problem,
            "model_type": self.config["model"]["type"],
            "seed": self.seed,
            "git_sha": _git_sha(),
            "dataset": self.config["data"]["source"],
            "split": self.config["split"],
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "metrics": metrics,
            "hyperparameters": self.config["model"].get("params", {}),
            "config_file": f"configs/{self.problem}.yaml",
            "mlflow_run_id": self.mlflow_run_id,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "note": (
                    "LightGBM/XGBoost have CPU-only wheels on macOS arm64 — no Metal "
                    "backend exists. Only src/models/autoencoder.py uses MPS."
                ),
            },
            "suspected_leakage": leak_warning,
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        reports_dir = Path(self.config["training"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / f"{self.problem}_metrics.csv"
        pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
            csv_path, index=False
        )
        logger.info("Wrote %s and %s", checkpoint_dir / "metadata.json", csv_path)

        return checkpoint_dir

    # ── helpers ──────────────────────────────────────────────────────────────

    def feature_bundle(self) -> dict[str, Any]:
        """Everything serving needs to rebuild the training feature frame.

        Written to ``features.joblib`` next to the model. Serving must never
        recompute any of it — see ``src/serving/preprocessing.py``.
        """
        return {
            "feature_columns": self.feature_columns,
            "artifacts": self.feature_artifacts,
            "category_dtypes": self.category_dtypes,
        }

    def _align(self, engineered: pd.DataFrame, *, capture: bool = False) -> pd.DataFrame:
        """Reindex to the exact training feature columns, in the exact order.

        Serving must produce a frame with these columns, in this order, with these
        category sets, or the model silently scores garbage. Doing it in one place
        is why ``tests/serving/test_skew.py`` can be a short assertion.

        Args:
            engineered: A frame from ``engineer_features``.
            capture: On the training split, record the categorical schema. On every
                other split, replay it.
        """
        aligned = engineered.reindex(columns=self.feature_columns)
        for column in aligned.columns:
            if str(aligned[column].dtype) == "object":
                aligned[column] = aligned[column].astype("category")

        if capture:
            self.category_dtypes = capture_category_dtypes(aligned)
        else:
            aligned = apply_category_dtypes(aligned, self.category_dtypes)
        return aligned

    def _assert_no_denylisted(self, denylist: list[str]) -> None:
        """Fail the run if any denylisted column survived feature selection."""
        leaked = sorted(set(denylist) & set(self.feature_columns))
        if leaked:
            raise ValueError(
                f"Denylisted columns reached the feature matrix: {leaked}. "
                f"This is the leak {self.problem}'s config exists to prevent. "
                f"Check {self.config['features']['module']}.get_feature_columns()."
            )

    def _check_expected_band(self, metrics: dict[str, float]) -> str | None:
        """Compare the measured ROC-AUC against the config's sanity band.

        Returns a warning string when the result is outside the band. A score
        *above* the band is the dangerous case: it means leakage, and the whole
        reason this repository was rebuilt was a metric nobody questioned.
        """
        expected = self.config.get("expected")
        if not expected:
            return None

        measured = metrics.get("test_roc_auc") or metrics.get("cv_roc_auc_mean")
        if measured is None:
            return None

        low, high = expected.get("roc_auc_min"), expected.get("roc_auc_max")
        if high is not None and measured > high:
            warning = (
                f"ROC-AUC {measured:.4f} EXCEEDS the expected ceiling {high}. "
                f"Treat this as suspected leakage, not success. Check for a random "
                f"split, an id column in the features, or a post-outcome field."
            )
            logger.error("SUSPECTED LEAKAGE: %s", warning)
            return warning
        if low is not None and measured < low:
            warning = (
                f"ROC-AUC {measured:.4f} is BELOW the expected floor {low}. "
                f"Check the split, the target construction and the feature count."
            )
            logger.warning("UNDERPERFORMING: %s", warning)
            return warning

        logger.info("ROC-AUC %.4f is inside the expected band [%s, %s].", measured, low, high)
        return None


def _positive_scores(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities from any estimator in the registry."""
    proba = model.predict_proba(matrix)
    return np.asarray(proba)[:, 1]


def _git_sha() -> str:
    """Current commit SHA, so a checkpoint can be traced to the code that made it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=get_project_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
