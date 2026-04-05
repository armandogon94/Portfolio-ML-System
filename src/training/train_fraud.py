"""Fraud detection model trainer (PyTorch Autoencoder + Isolation Forest baseline)."""

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.device import get_device
from src.evaluation.classification_metrics import compute_classification_metrics
from src.features.fraud_features import engineer_features, get_feature_columns
from src.models.fraud_autoencoder import FraudAutoencoder
from src.training.trainer import BaseTrainer, console


class FraudDetectionTrainer(BaseTrainer):

    def __init__(self, use_wandb: bool = True):
        super().__init__("fraud_detection", use_wandb=use_wandb)
        self.device = get_device()
        self.scaler = None
        self.threshold = None
        self.baseline_model = None
        console.print(f"   Using device: [bold]{self.device}[/bold]")

    def load_data(self) -> pd.DataFrame:
        path = self.config["data"]["raw_data_path"]
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame) -> dict:
        df, artifacts = engineer_features(df)
        feature_cols = get_feature_columns()
        target = self.config["features"]["target"]

        self.scaler = StandardScaler()

        # Split BEFORE scaling
        from sklearn.model_selection import train_test_split

        X = df[feature_cols].values.astype(np.float32)
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_seed"],
            stratify=y,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)

        # Autoencoder trains on NORMAL transactions only
        normal_mask = y_train == 0
        X_train_normal = X_train_scaled[normal_mask]

        return {
            "X_train_normal": X_train_normal,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "X_train_raw": X_train,
            "y_train": y_train,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "artifacts": artifacts,
        }

    def train(self, data: dict) -> None:
        params = self.config["model"]["params"]
        input_dim = data["X_train_normal"].shape[1]

        # --- Train Autoencoder ---
        console.print("   Training autoencoder on normal transactions...")
        self.model = FraudAutoencoder(
            input_dim=input_dim,
            hidden_dims=params.get("hidden_dims", [64, 32, 16]),
            dropout=params.get("dropout", 0.1),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.get("learning_rate", 0.001))
        criterion = torch.nn.MSELoss()

        train_tensor = torch.FloatTensor(data["X_train_normal"]).to(self.device)
        dataset = TensorDataset(train_tensor, train_tensor)
        loader = DataLoader(dataset, batch_size=params.get("batch_size", 256), shuffle=True)

        epochs = params.get("epochs", 50)
        patience = params.get("early_stopping_patience", 5)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            self.log_metric("train_loss", avg_loss, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                console.print(f"     Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

            if patience_counter >= patience:
                console.print(f"     Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(best_state)

        # Set anomaly threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            all_train = torch.FloatTensor(data["X_train_scaled"]).to(self.device)
            errors = self.model.reconstruction_error(all_train).cpu().numpy()

        percentile = params.get("threshold_percentile", 95)
        self.threshold = float(np.percentile(errors, percentile))
        console.print(f"   Anomaly threshold (p{percentile}): {self.threshold:.6f}")

        # --- Train Isolation Forest baseline ---
        console.print("   Training Isolation Forest baseline...")
        baseline_params = self.config["model"].get("baseline", {}).get("params", {})
        self.baseline_model = IsolationForest(
            contamination=baseline_params.get("contamination", 0.02),
            n_estimators=baseline_params.get("n_estimators", 100),
            random_state=baseline_params.get("random_state", 42),
        )
        self.baseline_model.fit(data["X_train_scaled"])

    def evaluate(self, data: dict) -> dict:
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(data["X_test_scaled"]).to(self.device)
            errors = self.model.reconstruction_error(X_test_tensor).cpu().numpy()

        # Autoencoder predictions
        y_pred_ae = (errors > self.threshold).astype(int)
        ae_metrics = compute_classification_metrics(
            data["y_test"], y_pred_ae, errors, prefix="test_autoencoder"
        )

        # Isolation Forest predictions
        iso_pred = self.baseline_model.predict(data["X_test_scaled"])
        y_pred_iso = (iso_pred == -1).astype(int)
        iso_scores = -self.baseline_model.score_samples(data["X_test_scaled"])
        iso_metrics = compute_classification_metrics(
            data["y_test"], y_pred_iso, iso_scores, prefix="test_isolation_forest"
        )

        return {**ae_metrics, **iso_metrics, "anomaly_threshold": self.threshold}

    def get_checkpoint_artifacts(self) -> dict:
        return {
            "autoencoder.pt": lambda path: torch.save(self.model.state_dict(), path),
            "scaler.pkl": lambda path: joblib.dump(self.scaler, path),
            "isolation_forest.pkl": lambda path: joblib.dump(self.baseline_model, path),
        }
