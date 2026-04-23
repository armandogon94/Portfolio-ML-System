"""Customer-churn trainer — XGBoost classifier on synthetic bank data."""

import pandas as pd

from src.evaluation.classification_metrics import compute_classification_metrics
from src.features.customer_churn_features import engineer_features, get_feature_columns
from src.models.credit_risk_model import create_model  # same XGB factory
from src.training.trainer import BaseTrainer


class CustomerChurnTrainer(BaseTrainer):
    """BaseTrainer subclass that trains an XGBoost churn classifier.

    Reuses :func:`src.models.credit_risk_model.create_model` because both
    models share the same binary-classification XGB surface; the factory is
    generic over its params dict.
    """

    def __init__(self, use_wandb: bool = True, modality: str | None = None):
        super().__init__("customer_churn", use_wandb=use_wandb, modality=modality)

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
            stratify=y,
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
            verbose=False,
        )

    def evaluate(self, data: dict) -> dict:
        y_prob = self.model.predict_proba(data["X_test"])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        return compute_classification_metrics(
            data["y_test"].values, y_pred, y_prob, prefix="test"
        )

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "model.json": lambda path: self.model.save_model(str(path)),
        }
