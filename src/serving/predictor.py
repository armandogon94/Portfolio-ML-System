"""Model predictor: loads checkpoints and runs inference for all models."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# IMPORTANT: Import xgboost BEFORE torch to avoid libomp segfault on macOS
import xgboost as xgb  # noqa: F401 - must be imported before torch
import lightgbm  # noqa: F401

import torch

from src.config import get_project_root
from src.features.credit_risk_features import engineer_features as credit_features
from src.features.credit_risk_features import get_feature_columns as credit_feature_cols
from src.features.fraud_features import get_feature_columns as fraud_feature_cols
from src.features.housing_features import engineer_features as housing_features
from src.features.housing_features import get_feature_columns as housing_feature_cols
from src.models.fraud_autoencoder import FraudAutoencoder
from src.models.lstm_forecaster import LSTMForecaster


def _get_device() -> torch.device:
    """Get best available device (avoids importing src.device which imports torch early)."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModelPredictor:
    """Unified inference interface for all trained models."""

    def __init__(self):
        self.root = get_project_root()
        self.device = _get_device()
        self._models = {}
        self._artifacts = {}

    def _ensure_loaded(self, problem: str) -> None:
        """Lazy-load a model checkpoint."""
        if problem in self._models:
            return

        checkpoint_dir = self.root / "checkpoints" / problem
        metadata_path = checkpoint_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"No checkpoint found for {problem}. Run training first.")

        with open(metadata_path) as f:
            metadata = json.load(f)

        self._artifacts[problem] = {"metadata": metadata}

        if problem == "credit_risk":
            model = xgb.XGBClassifier()
            model.load_model(str(checkpoint_dir / "model.json"))
            self._models[problem] = model

        elif problem == "fraud_detection":
            scaler = joblib.load(checkpoint_dir / "scaler.pkl")
            self._artifacts[problem]["scaler"] = scaler

            input_dim = len(fraud_feature_cols())
            autoencoder = FraudAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16])
            autoencoder.load_state_dict(
                torch.load(checkpoint_dir / "autoencoder.pt", weights_only=True, map_location=self.device)
            )
            autoencoder.to(self.device)
            autoencoder.eval()
            self._models[problem] = autoencoder

            iso = joblib.load(checkpoint_dir / "isolation_forest.pkl")
            self._artifacts[problem]["isolation_forest"] = iso
            self._artifacts[problem]["threshold"] = metadata["metrics"].get("anomaly_threshold", 0.18)

        elif problem == "price_prediction":
            model = joblib.load(checkpoint_dir / "model.pkl")
            self._models[problem] = model

        elif problem == "demand_forecasting":
            scalers = joblib.load(checkpoint_dir / "scalers.pkl")
            self._artifacts[problem]["scalers"] = scalers

            model = LSTMForecaster(
                input_size=1, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=7,
            )
            model.load_state_dict(
                torch.load(checkpoint_dir / "lstm.pt", weights_only=True, map_location=self.device)
            )
            model.to(self.device)
            model.eval()
            self._models[problem] = model

    def predict_credit_risk(self, data: dict) -> dict:
        """Score a loan application."""
        self._ensure_loaded("credit_risk")
        model = self._models["credit_risk"]

        df = pd.DataFrame([data])
        df = credit_features(df)
        features = df[credit_feature_cols()]

        prob = model.predict_proba(features)[0][1]
        risk_score = float(prob)

        if risk_score < 0.15:
            recommendation = "APPROVE"
        elif risk_score < 0.40:
            recommendation = "REVIEW"
        else:
            recommendation = "DECLINE"

        return {
            "risk_score": risk_score,
            "recommendation": recommendation,
            "confidence": float(max(risk_score, 1 - risk_score)),
            "default_probability": risk_score,
        }

    def predict_fraud(self, data: dict) -> dict:
        """Detect fraud in a transaction."""
        self._ensure_loaded("fraud_detection")

        from src.features.fraud_features import engineer_features

        df = pd.DataFrame([data])
        df, _ = engineer_features(df)
        feature_cols = fraud_feature_cols()
        X = df[feature_cols].values.astype(np.float32)

        scaler = self._artifacts["fraud_detection"]["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)

        model = self._models["fraud_detection"]
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            recon_error = float(model.reconstruction_error(X_tensor).cpu().item())

        threshold = self._artifacts["fraud_detection"]["threshold"]
        is_anomaly = recon_error > threshold

        iso = self._artifacts["fraud_detection"]["isolation_forest"]
        iso_pred = iso.predict(X_scaled)[0]
        iso_score = float(-iso.score_samples(X_scaled)[0])

        fraud_prob = min(recon_error / (threshold * 3), 1.0)
        if fraud_prob < 0.2:
            risk_level = "LOW"
        elif fraud_prob < 0.5:
            risk_level = "MEDIUM"
        elif fraud_prob < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "reconstruction_error": recon_error,
            "anomaly_threshold": threshold,
            "is_anomaly_autoencoder": bool(is_anomaly),
            "is_anomaly_isolation_forest": bool(iso_pred == -1),
            "isolation_forest_score": iso_score,
        }

    def predict_price(self, data: dict) -> dict:
        """Predict real estate price."""
        self._ensure_loaded("price_prediction")
        model = self._models["price_prediction"]

        df = pd.DataFrame([data])
        df = housing_features(df)
        features = df[housing_feature_cols()]

        prediction = float(model.predict(features)[0])

        return {
            "predicted_price": prediction,
            "price_range_low": prediction * 0.90,
            "price_range_high": prediction * 1.10,
        }

    def predict_demand(self, product: str, recent_demand: list[float] | None = None) -> dict:
        """Forecast demand for a product category."""
        self._ensure_loaded("demand_forecasting")

        model = self._models["demand_forecasting"]
        scalers = self._artifacts["demand_forecasting"]["scalers"]

        if product not in scalers:
            available = list(scalers.keys())
            raise ValueError(f"Unknown product '{product}'. Available: {available}")

        scaler = scalers[product]

        if recent_demand is None:
            raw_path = self.root / "data" / "raw" / "daily_demand.csv"
            df = pd.read_csv(raw_path, parse_dates=["date"])
            product_df = df[df["product_category"] == product].sort_values("date")
            recent_demand = product_df["demand"].tail(30).tolist()

        values = np.array(recent_demand, dtype=float).reshape(-1, 1)
        scaled = scaler.transform(values).flatten()

        if len(scaled) < 30:
            scaled = np.pad(scaled, (30 - len(scaled), 0), mode="edge")
        scaled = scaled[-30:]

        X = torch.FloatTensor(scaled.reshape(1, 30, 1)).to(self.device)

        with torch.no_grad():
            pred_scaled = model(X).cpu().numpy().flatten()

        predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        return {
            "product": product,
            "forecast_days": 7,
            "predictions": [float(p) for p in predictions],
            "avg_predicted_demand": float(np.mean(predictions)),
        }

    def get_model_info(self) -> dict:
        """Return metadata for all available models."""
        info = {}
        for problem in ["credit_risk", "fraud_detection", "price_prediction", "demand_forecasting"]:
            metadata_path = self.root / "checkpoints" / problem / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    info[problem] = json.load(f)
        return info
