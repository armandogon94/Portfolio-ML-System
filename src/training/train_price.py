"""Price prediction model trainer.

Phase A.1.7: supports three data modalities (synthetic | stream | mixed)
via :func:`src.data.modality.load_for_modality`. The modality is passed at
``__init__`` time — when ``None`` (default) the legacy synthetic-only path
is used to preserve backward compatibility with older invocations.
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.evaluation.regression_metrics import compute_regression_metrics
from src.features.housing_features import engineer_features, get_feature_columns
from src.models.price_model import create_model
from src.training.trainer import BaseTrainer


class PricePredictionTrainer(BaseTrainer):

    def __init__(self, use_wandb: bool = True, modality: str | None = None):
        super().__init__("price_prediction", use_wandb=use_wandb, modality=modality)

    def load_data(self) -> pd.DataFrame:
        """Load housing data according to the selected modality.

        - modality=None:         legacy synthetic-only read from raw_data_path
        - modality='synthetic':  same as None but via the modality dispatcher
        - modality='stream':     Kaggle Zillow dataset adapted to canonical schema
        - modality='mixed':      synthetic + stream concatenated with a 'modality'
                                 feature column indicating the row origin
        """
        raw_path = self.config["data"]["raw_data_path"]

        # Legacy path: no modality → just read the synthetic CSV directly.
        if self.modality is None:
            return pd.read_csv(raw_path)

        # Dispatcher path: synthetic / stream / mixed.
        from src.data.adapters import housing_adapter
        from src.data.modality import load_for_modality

        df = load_for_modality(
            self.modality,
            synthetic_loader=lambda: pd.read_csv(raw_path),
            stream_slug=self.config["data"].get("kaggle_slug"),
            stream_file=self.config["data"].get("stream_file"),
            stream_adapter=housing_adapter,
        )

        # 'mixed' adds a 'modality' column which isn't a housing feature and
        # isn't in get_feature_columns(); drop it before preprocessing so the
        # feature pipeline stays consistent across modalities.
        if "modality" in df.columns:
            df = df.drop(columns=["modality"])

        return df

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
        self.model = create_model(params)

        self.model.fit(
            data["X_train"], data["y_train"],
            eval_set=[(data["X_test"], data["y_test"])],
            callbacks=[],
        )

    def evaluate(self, data: dict) -> dict:
        y_pred = self.model.predict(data["X_test"])
        return compute_regression_metrics(data["y_test"].values, y_pred, prefix="test")

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "model.pkl": lambda path: joblib.dump(self.model, path),
        }
