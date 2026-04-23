"""H-1B visa approval model trainer — XGBoost binary classifier."""

import pandas as pd

from src.evaluation.classification_metrics import compute_classification_metrics
from src.features.h1b_approval_features import engineer_features, get_feature_columns
from src.models.credit_risk_model import create_model
from src.training.trainer import BaseTrainer


class H1BApprovalTrainer(BaseTrainer):
    """Trains ``h1b_approval.yaml`` — XGBoost classifier on LCA-inspired data.

    The model shape is identical to credit_risk (binary XGBoost), so we reuse
    ``src.models.credit_risk_model.create_model`` rather than duplicating the
    factory. The only H-1B-specific pieces are feature engineering + the
    target column.
    """

    def __init__(self, use_wandb: bool = True, modality: str | None = None):
        super().__init__("h1b_approval", use_wandb=use_wandb, modality=modality)

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
        self.model = create_model(params)
        self.model.fit(
            data["X_train"],
            data["y_train"],
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
