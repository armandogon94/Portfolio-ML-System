"""One config-driven trainer for all three problems.

This file replaces the eight ``src/training/train_<problem>.py`` modules that
preceded it. They were 80% identical and diverged in the 20% that mattered: two of
them applied a different split, one silently skipped early stopping. A single
implementation driven by ``configs/<problem>.yaml`` makes "config-driven training"
a fact about the repository rather than a claim in its README.

What the config controls end to end: the dataset and adapter, the feature module,
the split strategy and key, the model and its hyperparameters, the baselines, the
reported metrics, the seed, and the expected-range sanity band used as a smoke
alarm.

Guardrails that are code, not documentation:

* ``--sample`` opens no tracking run and refuses checkpoint/CSV writes. Fixture
  numbers therefore cannot enter MLflow or become published numbers by accident.
* Feature-fitting state (frequency maps, group means) is fitted on train only and
  reused verbatim for validation, test and serving.
* Every denylisted column is dropped before the model sees the frame, and the drop
  is asserted rather than assumed.
* The measured metric is compared against ``sanity_band``. That heuristic is
  recorded as a warning, never presented as a leakage test; the denylist and
  split-column exclusion are the actual leakage controls.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
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

_DEGENERATE_SCORE_STD_MAX = 1e-12


class TabularTrainer(BaseTrainer):
    """Train, evaluate and checkpoint one problem, entirely from its config.

    Args:
        config_name: A problem name resolving to ``configs/<name>.yaml``.
        use_wandb: Enable W&B logging when ``WANDB_API_KEY`` is set.
        sample: Train on the committed CI fixture instead of the real dataset.
            Opens no tracking run and **forces checkpointing off** — see
            :meth:`save_artifacts`.
    """

    def __init__(
        self,
        config_name: str,
        *,
        use_wandb: bool = False,
        sample: bool = False,
        tracking_model_type: str | None = None,
        registry_name: str | None = None,
    ):
        # BaseTrainer needs sample before it considers opening an MLflow run.
        super().__init__(
            config_name,
            use_wandb=use_wandb,
            sample=sample,
            tracking_model_type=tracking_model_type,
            registry_name=registry_name,
        )
        self.feature_columns: list[str] = []
        self.feature_artifacts: dict[str, Any] = {}
        #: Exact training category sets, replayed at serving time. See
        #: src/features/schema.py for why permuted codes are the real danger.
        self.category_dtypes: dict[str, list] = {}
        self.baseline_metrics: dict[str, dict[str, float]] = {}
        self._fold_metrics: list[dict[str, float]] = []
        self._last_scores: dict[str, np.ndarray] = {}
        self._oof_predictions: list[pd.DataFrame] = []
        self.checkpoint_fit: dict[str, Any] = {}

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
        frame leaks information about the test distribution into training.
        """
        features = importlib.import_module(self.config["features"]["module"])
        target = self.config["data"]["target"]
        denylist = list(self.config["data"].get("denylist", [])) or [target]
        split_column = self.config["split"].get("column")
        if split_column and split_column not in denylist:
            denylist.append(split_column)

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
        model_scores = _positive_scores(model, data["X_test"])
        self._last_scores = {"model": model_scores}
        metrics = compute_classification_metrics(data["y_test"], model_scores, prefix="test")

        for baseline in self.config.get("baselines", []):
            name = baseline["type"]
            estimator = create_model(name, baseline.get("params"), seed=self.seed)
            estimator.fit(data["X_train"], data["y_train"])
            baseline_scores = _positive_scores(estimator, data["X_test"])
            self._last_scores[name] = baseline_scores
            scored = compute_classification_metrics(
                data["y_test"],
                baseline_scores,
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
        self._fold_metrics = []
        self._oof_predictions = []
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
            if len(splits) > 1:
                self._record_oof_predictions(split, data, fold=index + 1)
            metrics = fold_metrics

        if len(splits) > 1:
            # The CV aggregate is the report. A single fold is deliberately not
            # retained under test_* names where a caller could mistake it for the
            # headline result; per-row fold evidence lives in the OOF CSV.
            metrics = aggregate_folds(self._fold_metrics, prefix="cv")
            if self.config["training"].get("refit_full_after_cv", True):
                model = self._refit_on_all_rows(frame)
                self.checkpoint_fit = {
                    "scope": "all_rows_refit_after_cross_validation",
                    "n_rows": len(frame),
                    "evaluation": "out_of_fold_predictions",
                }
            else:
                assert model is not None
                n_rows = len(data["X_train"])
                setattr(model, "training_rows_", n_rows)
                self.checkpoint_fit = {
                    "scope": f"final_cross_validation_fold_{len(splits)}",
                    "n_rows": n_rows,
                    "evaluation": "cross_validation_mean_and_standard_deviation",
                    "note": "Refit disabled explicitly in training.refit_full_after_cv.",
                }
        else:
            assert model is not None
            n_rows = len(data["X_train"])
            setattr(model, "training_rows_", n_rows)
            self.checkpoint_fit = {
                "scope": "training_partition",
                "n_rows": n_rows,
                "evaluation": "held_out_test_partition",
            }

        metrics["n_features"] = float(len(self.feature_columns))
        metrics["training_time_seconds"] = round(time.time() - started, 1)
        self.metrics = metrics

        sanity_band_warning = self._check_sanity_band(metrics)
        checkpoint_dir = self.save_artifacts(
            model, metrics, sanity_band_warning=sanity_band_warning
        )
        # Local artifacts are durable before any network/file-store tracker is
        # allowed to fail. This ordering implements ADR-0002.
        self.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        if checkpoint_dir is not None:
            self.register_model(str(checkpoint_dir))
        self.finish()
        return metrics

    def save_artifacts(
        self,
        model: Any,
        metrics: dict[str, float],
        *,
        sanity_band_warning: str | None,
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
            # ISO-8601 UTC. Read by the dashboard's "last trained" column and by
            # anyone asking whether a published number predates a code change.
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.config["data"]["source"],
            "split": self.config["split"],
            "feature_columns": self.feature_columns,
            "n_features": len(self.feature_columns),
            "metrics": metrics,
            "hyperparameters": self.config["model"].get("params", {}),
            "config_file": f"configs/{self.config_name}.yaml",
            "mlflow_run_id": self.mlflow_run_id,
            "checkpoint_fit": self.checkpoint_fit,
            "evaluation_predictions": (
                {
                    "kind": "out_of_fold",
                    "path": f"reports/{self.config_name}_oof_predictions.csv",
                    "note": "Each row was scored only by the fold that held it out.",
                }
                if self._oof_predictions
                else {"kind": "held_out_test_partition"}
            ),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "note": (
                    "LightGBM/XGBoost have CPU-only wheels on macOS arm64 — no Metal "
                    "backend exists. Only src/models/autoencoder.py uses MPS."
                ),
            },
            "leakage_controls": {
                "denylist_enforced": True,
                "split_column_excluded": True,
                "sanity_band_is_smoke_alarm_only": True,
            },
            "sanity_band_warning": sanity_band_warning,
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        reports_dir = Path(self.config["training"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / f"{self.config_name}_metrics.csv"
        pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(metrics.items())]).to_csv(
            csv_path, index=False
        )
        if self._oof_predictions:
            oof_path = reports_dir / f"{self.config_name}_oof_predictions.csv"
            pd.concat(self._oof_predictions, ignore_index=True).sort_values("row_index").to_csv(
                oof_path, index=False
            )
            logger.info("Wrote held-out OOF predictions to %s", oof_path)
        logger.info("Wrote %s and %s", checkpoint_dir / "metadata.json", csv_path)

        return checkpoint_dir

    def _record_oof_predictions(self, split: Split, data: dict[str, Any], *, fold: int) -> None:
        """Collect held-out scores for one CV fold, preserving source row ids."""
        rows: dict[str, Any] = {
            "row_index": split.test,
            "fold": fold,
            "y_true": data["y_test"],
        }
        rows.update({f"score_{name}": values for name, values in self._last_scores.items()})
        self._oof_predictions.append(pd.DataFrame(rows))

    def _refit_on_all_rows(self, frame: pd.DataFrame) -> Any:
        """Fit the checkpoint estimator on all rows after CV evaluation."""
        all_rows = np.arange(len(frame), dtype=int)
        full = Split(
            train=all_rows,
            val=np.array([], dtype=int),
            test=np.array([], dtype=int),
        )
        data = self.build_matrix(frame, full)
        model = self.train(data)
        setattr(model, "training_rows_", len(data["X_train"]))
        logger.info(
            "Refit final %s estimator on all %d rows after cross-validation.",
            self.config["model"]["type"],
            len(data["X_train"]),
        )
        return model

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

    def _check_sanity_band(self, metrics: dict[str, float]) -> str | None:
        """Run the degenerate-score gate and expected-range smoke alarm.

        The configured band is not a leakage test. It only says that a result is
        surprising enough to investigate. Leakage is prevented structurally by
        time-aware splits, the denylist, and split-column exclusion.
        """
        is_cv = self.config["split"]["type"] == "stratified_kfold"
        score_std = metrics.get("cv_score_std_mean") if is_cv else metrics.get("test_score_std")
        if score_std is not None and score_std <= _DEGENERATE_SCORE_STD_MAX:
            warning = (
                f"DEGENERATE predictions: score standard deviation is {score_std:.4g}. "
                "The model returned a constant score, so ranking metrics are not "
                "evidence. Check the fitted estimator and positive-class score path."
            )
            logger.error("QUALITY GATE FAILED: %s", warning)
            return warning

        band = self.config.get("sanity_band")
        if not band:
            return None

        metric = band["metric"]
        label = "PR-AUC" if metric == "pr_auc" else "ROC-AUC"
        measured = metrics.get(f"cv_{metric}_mean") if is_cv else metrics.get(f"test_{metric}")
        if measured is None:
            return None

        low, high = band.get("min"), band.get("max")
        if high is not None and measured > high:
            warning = (
                f"{label} {measured:.4f} is ABOVE the expected-range sanity band "
                f"maximum {high}. This smoke alarm cannot diagnose leakage; "
                f"investigate the split, feature set, and target before publishing."
            )
            logger.warning("SANITY BAND WARNING: %s", warning)
            return warning
        if low is not None and measured < low:
            warning = (
                f"{label} {measured:.4f} is BELOW the expected-range sanity band "
                f"minimum {low}. "
                f"Check the split, the target construction and the feature count."
            )
            logger.warning("SANITY BAND WARNING: %s", warning)
            return warning

        logger.info(
            "%s %.4f is inside the expected-range sanity band [%s, %s].",
            label,
            measured,
            low,
            high if high is not None else "unbounded",
        )
        return None


def _positive_scores(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities from any estimator in the registry.

    A one-column ``predict_proba`` means the estimator only ever saw one class,
    which on a time split means every positive landed outside the training
    partition. That produces meaningless metrics, so it fails here with the
    cause named rather than as an ``IndexError`` four frames deeper.
    """
    proba = np.asarray(model.predict_proba(matrix))
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(
            "The estimator was fitted on a single class, so there is no "
            "positive-class probability to score. The training partition "
            "contains no positive rows — check the split boundaries and the "
            "positive rate of the data this run was given."
        )
    return proba[:, 1]


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
