"""Model predictor: loads checkpoints and runs inference for all models."""

import json
import os

import joblib
import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
import torch

# IMPORTANT: KMP_DUPLICATE_LIB_OK=TRUE must be set before importing these
import xgboost as xgb  # noqa: F401 - import early to avoid libomp conflict on macOS

from src.config import get_project_root
from src.explainability.gradient_explainer import GradientExplainer
from src.explainability.shap_explainer import SHAPExplainer
from src.features.credit_risk_features import engineer_features as credit_features
from src.features.credit_risk_features import get_feature_columns as credit_feature_cols
from src.features.customer_churn_features import engineer_features as churn_features
from src.features.customer_churn_features import get_feature_columns as churn_feature_cols
from src.features.delivery_eta_features import engineer_features as delivery_eta_features
from src.features.delivery_eta_features import (
    get_feature_columns as delivery_eta_feature_cols,
)
from src.features.dental_noshow_features import engineer_features as dental_noshow_features
from src.features.dental_noshow_features import get_feature_columns as dental_noshow_feature_cols
from src.features.fraud_features import get_feature_columns as fraud_feature_cols
from src.features.h1b_approval_features import engineer_features as h1b_features
from src.features.h1b_approval_features import get_feature_columns as h1b_feature_cols
from src.features.heart_disease_features import engineer_features as heart_features
from src.features.heart_disease_features import get_feature_columns as heart_feature_cols
from src.features.housing_features import engineer_features as housing_features
from src.features.housing_features import get_feature_columns as housing_feature_cols
from src.features.rental_price_features import engineer_features as rental_features
from src.features.rental_price_features import get_feature_columns as rental_feature_cols
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

        elif problem == "dental_noshow":
            model = xgb.XGBClassifier()
            model.load_model(str(checkpoint_dir / "model.json"))
            self._models[problem] = model

        elif problem == "customer_churn":
            model = xgb.XGBClassifier()
            model.load_model(str(checkpoint_dir / "model.json"))
            self._models[problem] = model

        elif problem == "fraud_detection":
            scaler = joblib.load(checkpoint_dir / "scaler.pkl")
            self._artifacts[problem]["scaler"] = scaler

            input_dim = len(fraud_feature_cols())
            autoencoder = FraudAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16])
            autoencoder.load_state_dict(
                torch.load(
                    checkpoint_dir / "autoencoder.pt", weights_only=True, map_location=self.device
                )
            )
            autoencoder.to(self.device)
            autoencoder.eval()
            self._models[problem] = autoencoder

            iso = joblib.load(checkpoint_dir / "isolation_forest.pkl")
            self._artifacts[problem]["isolation_forest"] = iso
            self._artifacts[problem]["threshold"] = metadata["metrics"].get(
                "anomaly_threshold", 0.18
            )

        elif problem == "price_prediction":
            model = joblib.load(checkpoint_dir / "model.pkl")
            self._models[problem] = model

        elif problem == "rental_price":
            model = joblib.load(checkpoint_dir / "model.pkl")
            self._models[problem] = model

        elif problem == "heart_disease":
            # LightGBM classifier serialised via joblib — same shape as price.
            model = joblib.load(checkpoint_dir / "model.pkl")
            self._models[problem] = model

        elif problem == "delivery_eta":
            model = xgb.XGBRegressor()
            model.load_model(str(checkpoint_dir / "model.json"))
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

        elif problem == "h1b_approval":
            model = xgb.XGBClassifier()
            model.load_model(str(checkpoint_dir / "model.json"))
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

    def predict_delivery_eta(self, data: dict) -> dict:
        """Predict delivery ETA in hours with a ±20% confidence band."""
        self._ensure_loaded("delivery_eta")
        model = self._models["delivery_eta"]

        df = pd.DataFrame([data])
        df = delivery_eta_features(df)
        features = df[delivery_eta_feature_cols()]

        eta = float(model.predict(features)[0])

        return {
            "eta_hours": eta,
            # Heuristic band — Phase A.7 spec: [eta*0.8, eta*1.2].
            "confidence_interval": [eta * 0.8, eta * 1.2],
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
        """Return metadata for every checkpoint under ``checkpoints/``.

        Dynamically scans ``self.root / "checkpoints" / <problem> / metadata.json``
        — adding a new model checkpoint dir requires no code change here.
        Returns ``{}`` when the checkpoints directory is missing or empty.
        Malformed ``metadata.json`` files are skipped silently rather than
        crashing the endpoint.
        """
        info: dict = {}
        checkpoints_root = self.root / "checkpoints"
        if not checkpoints_root.is_dir():
            return info
        for metadata_path in sorted(checkpoints_root.glob("*/metadata.json")):
            problem = metadata_path.parent.name
            try:
                with open(metadata_path) as f:
                    info[problem] = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Skip partial/corrupt checkpoints rather than crash the endpoint.
                continue
        return info

    def explain_credit_risk(self, data: dict) -> dict:
        """Explain a credit risk prediction with SHAP values."""
        self._ensure_loaded("credit_risk")
        model = self._models["credit_risk"]

        df = pd.DataFrame([data])
        df = credit_features(df)
        features = df[credit_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, credit_feature_cols())

    def explain_price(self, data: dict) -> dict:
        """Explain a price prediction with SHAP values."""
        self._ensure_loaded("price_prediction")
        model = self._models["price_prediction"]

        df = pd.DataFrame([data])
        df = housing_features(df)
        features = df[housing_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, housing_feature_cols())

    def predict_rental_price(self, data: dict) -> dict:
        """Predict nightly rental rate (A.3 — Real Estate).

        Returns the predicted rate plus a simple ±10% confidence interval.
        The band is a demo-grade heuristic (same shape as ``predict_price``)
        — a future slice can replace it with LightGBM quantile regressors
        (``objective=quantile`` at alpha=0.1 / 0.9) when we want a model-
        derived interval.
        """
        self._ensure_loaded("rental_price")
        model = self._models["rental_price"]

        df = pd.DataFrame([data])
        df = rental_features(df)
        features = df[rental_feature_cols()]

        prediction = float(model.predict(features)[0])

        low = round(prediction * 0.90, 2)
        high = round(prediction * 1.10, 2)
        return {
            "predicted_rate": round(prediction, 2),
            "confidence_interval": [low, high],
        }

    def explain_rental_price(self, data: dict) -> dict:
        """Explain a rental-price prediction with SHAP values."""
        self._ensure_loaded("rental_price")
        model = self._models["rental_price"]

        df = pd.DataFrame([data])
        df = rental_features(df)
        features = df[rental_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, rental_feature_cols())

    def explain_delivery_eta(self, data: dict) -> dict:
        """Explain a delivery-ETA prediction with SHAP values."""
        self._ensure_loaded("delivery_eta")
        model = self._models["delivery_eta"]

        df = pd.DataFrame([data])
        df = delivery_eta_features(df)
        features = df[delivery_eta_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, delivery_eta_feature_cols())

    def explain_fraud(self, data: dict) -> dict:
        """Explain a fraud prediction with gradient-based feature importance."""
        self._ensure_loaded("fraud_detection")

        from src.features.fraud_features import engineer_features

        df = pd.DataFrame([data])
        df, _ = engineer_features(df)
        feature_cols = fraud_feature_cols()
        X = df[feature_cols].values.astype(np.float32)

        scaler = self._artifacts["fraud_detection"]["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)

        model = self._models["fraud_detection"]
        X_tensor = torch.FloatTensor(X_scaled)

        explainer = GradientExplainer()
        return explainer.explain(model, X_tensor, feature_cols)

    def predict_dental_noshow(self, data: dict) -> dict:
        """Predict the probability that a dental patient misses their appointment."""
        self._ensure_loaded("dental_noshow")
        model = self._models["dental_noshow"]

        df = pd.DataFrame([data])
        df = dental_noshow_features(df)
        features = df[dental_noshow_feature_cols()]

        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.40:
            risk_band = "HIGH_RISK"
        elif prob <= 0.15:
            risk_band = "LIKELY_TO_SHOW"
        else:
            risk_band = "MODERATE"

        return {
            "probability_no_show": prob,
            "risk_band": risk_band,
            "confidence": float(max(prob, 1 - prob)),
        }

    def explain_dental_noshow(self, data: dict) -> dict:
        """Explain a dental no-show prediction with SHAP values."""
        self._ensure_loaded("dental_noshow")
        model = self._models["dental_noshow"]

        df = pd.DataFrame([data])
        df = dental_noshow_features(df)
        features = df[dental_noshow_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, dental_noshow_feature_cols())

    def predict_heart_disease(self, data: dict) -> dict:
        """Score cardiac patient vitals for heart disease probability.

        Output shape:
            - probability_disease: float in [0, 1]
            - risk_band: "HIGH" if prob >= 0.5, "ELEVATED" if >= 0.25, else "LOW"
            - confidence: float in [0, 1] (distance from decision boundary)
        """
        self._ensure_loaded("heart_disease")
        model = self._models["heart_disease"]

        df = pd.DataFrame([data])
        df = heart_features(df)
        features = df[heart_feature_cols()]

        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.5:
            risk_band = "HIGH"
        elif prob >= 0.25:
            risk_band = "ELEVATED"
        else:
            risk_band = "LOW"

        return {
            "probability_disease": prob,
            "risk_band": risk_band,
            "confidence": float(max(prob, 1 - prob)),
        }

    def explain_heart_disease(self, data: dict) -> dict:
        """Explain a heart-disease prediction with SHAP values (LightGBM)."""
        self._ensure_loaded("heart_disease")
        model = self._models["heart_disease"]

        df = pd.DataFrame([data])
        df = heart_features(df)
        features = df[heart_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, heart_feature_cols())

    def predict_customer_churn(self, data: dict) -> dict:
        """Score a bank customer for churn risk.

        Thresholds mirror the credit-risk tiering convention but invert the
        semantic: high probability means "act now, the customer is leaving".
        The ≥0.6 and ≥0.3 cutoffs were chosen to keep URGENT_OUTREACH at
        roughly the top decile while surfacing a middle band for PROACTIVE
        check-ins, which matches how retention ops queues are typically
        structured.
        """
        self._ensure_loaded("customer_churn")
        model = self._models["customer_churn"]

        df = pd.DataFrame([data])
        df = churn_features(df)
        features = df[churn_feature_cols()]

        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.6:
            recommendation = "URGENT_OUTREACH"
        elif prob >= 0.3:
            recommendation = "PROACTIVE_CHECKIN"
        else:
            recommendation = "NO_ACTION"

        return {
            "probability_churn": prob,
            "retention_recommendation": recommendation,
            "confidence": float(max(prob, 1 - prob)),
        }

    def explain_customer_churn(self, data: dict) -> dict:
        """Explain a churn prediction with SHAP values (TreeExplainer)."""
        self._ensure_loaded("customer_churn")
        model = self._models["customer_churn"]

        df = pd.DataFrame([data])
        df = churn_features(df)
        features = df[churn_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, churn_feature_cols())

    def predict_h1b_approval(self, data: dict) -> dict:
        """Score an H-1B petition. Returns approval probability + recommendation.

        Tier thresholds (matches the Phase A.8 spec):
        - ``probability_approval >= 0.7`` → APPROVE_LIKELY
        - ``0.4 <= probability_approval < 0.7`` → REVIEW
        - ``probability_approval < 0.4`` → DECLINE_LIKELY
        """
        self._ensure_loaded("h1b_approval")
        model = self._models["h1b_approval"]

        df = pd.DataFrame([data])
        df = h1b_features(df)
        features = df[h1b_feature_cols()]

        prob = float(model.predict_proba(features)[0][1])

        if prob >= 0.7:
            recommendation = "APPROVE_LIKELY"
        elif prob >= 0.4:
            recommendation = "REVIEW"
        else:
            recommendation = "DECLINE_LIKELY"

        return {
            "probability_approval": prob,
            "recommendation": recommendation,
            "confidence": float(max(prob, 1 - prob)),
        }

    def explain_h1b_approval(self, data: dict) -> dict:
        """Explain an H-1B prediction with SHAP values (XGBoost TreeExplainer)."""
        self._ensure_loaded("h1b_approval")
        model = self._models["h1b_approval"]

        df = pd.DataFrame([data])
        df = h1b_features(df)
        features = df[h1b_feature_cols()]

        explainer = SHAPExplainer()
        return explainer.explain(model, features, h1b_feature_cols())
