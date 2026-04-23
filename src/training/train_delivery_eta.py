"""Delivery ETA regression trainer (Phase A.7 — Logistics).

XGBoost regressor on the synthetic delivery dataset. Same BaseTrainer shape as
``train_credit_risk`` (XGBoost) and ``train_price`` (regression) — the only
differences are the objective (``reg:squarederror``), the metric family
(regression, not classification), and the pass-through feature pipeline.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from src.evaluation.regression_metrics import compute_regression_metrics
from src.features.delivery_eta_features import engineer_features, get_feature_columns
from src.training.trainer import BaseTrainer


class DeliveryEtaTrainer(BaseTrainer):
    """Train the delivery-ETA XGBoost regressor end-to-end."""

    def __init__(self, use_wandb: bool = True, modality: str | None = None):
        super().__init__("delivery_eta", use_wandb=use_wandb, modality=modality)

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
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_seed"],
        )

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }

    def train(self, data: dict) -> None:
        params = self.config["model"]["params"]
        # XGBRegressor — mirrors the credit_risk_model factory for classifiers.
        self.model = xgb.XGBRegressor(
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=params.get("n_estimators", 200),
            objective=params.get("objective", "reg:squarederror"),
            eval_metric=params.get("eval_metric", "rmse"),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_test"], data["y_test"])],
            verbose=False,
        )

    def evaluate(self, data: dict) -> dict:
        y_pred = self.model.predict(data["X_test"])
        return compute_regression_metrics(
            data["y_test"].values, y_pred, prefix="test"
        )

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "model.json": lambda path: self.model.save_model(str(path)),
        }
