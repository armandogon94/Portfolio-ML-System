"""Heart disease risk model trainer (LightGBM binary classifier).

Mirrors :class:`CreditRiskTrainer` in structure but uses LightGBM so the
saved artefact is `model.pkl` via joblib (same loading pattern as
``predict_price``). Target is ``disease`` (0/1). Metrics are reported
via :func:`compute_classification_metrics`, so downstream tooling that
keys on ``test_auc_roc`` (e.g. ``RECOMMENDATION_KEY`` in scripts/train.py)
works without changes.
"""

from __future__ import annotations

import joblib
import lightgbm as lgb
import pandas as pd

from src.evaluation.classification_metrics import compute_classification_metrics
from src.features.heart_disease_features import engineer_features, get_feature_columns
from src.training.trainer import BaseTrainer


class HeartDiseaseTrainer(BaseTrainer):
    """LightGBM classifier for synthetic Cleveland-style heart disease data."""

    def __init__(self, use_wandb: bool = True, modality: str | None = None):
        super().__init__("heart_disease", use_wandb=use_wandb, modality=modality)

    def load_data(self) -> pd.DataFrame:
        path = self.config["data"]["raw_data_path"]
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame) -> dict:
        df = engineer_features(df)
        feature_cols = get_feature_columns()
        target = self.config["features"]["target"]

        from sklearn.model_selection import train_test_split

        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_seed"],
            stratify=y,
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def train(self, data: dict) -> None:
        params = self.config["model"]["params"]
        self.model = lgb.LGBMClassifier(
            objective=params.get("objective", "binary"),
            metric=params.get("metric", "auc"),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", -1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            verbose=-1,
            random_state=42,
        )

        self.model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_test"], data["y_test"])],
            callbacks=[],
        )

    def evaluate(self, data: dict) -> dict:
        y_prob = self.model.predict_proba(data["X_test"])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        return compute_classification_metrics(
            data["y_test"].values, y_pred, y_prob, prefix="test"
        )

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "model.pkl": lambda path: joblib.dump(self.model, path),
        }
