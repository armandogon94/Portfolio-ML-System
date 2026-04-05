"""Price prediction model trainer."""

import joblib
import pandas as pd

from src.features.housing_features import engineer_features, get_feature_columns
from src.models.price_model import create_model
from src.evaluation.regression_metrics import compute_regression_metrics
from src.training.trainer import BaseTrainer


class PricePredictionTrainer(BaseTrainer):

    def __init__(self, use_wandb: bool = True):
        super().__init__("price_prediction", use_wandb=use_wandb)

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
