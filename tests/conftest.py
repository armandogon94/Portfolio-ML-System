"""Test configuration and shared fixtures.

Provides tiny trained model fixtures for fast, deterministic tests.
"""

# CRITICAL: Set before ANY imports — prevents libomp segfault when
# xgboost/lightgbm and PyTorch coexist on macOS
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
from unittest.mock import patch as _mock_patch

import joblib
import lightgbm as lgb  # noqa: F401
import numpy as np
import pandas as pd
import pytest
import torch
import xgboost as xgb  # noqa: F401 — import before torch

# Force CPU for entire test session — MPS operations hang in pytest on macOS
_mps_patcher = _mock_patch("torch.backends.mps.is_available", return_value=False)
_mps_patcher.start()

from src.features.credit_risk_features import (
    engineer_features as credit_features,
    get_feature_columns as credit_feature_cols,
)
from src.features.fraud_features import (
    engineer_features as fraud_engineer,
    get_feature_columns as fraud_feature_cols,
)
from src.features.housing_features import (
    engineer_features as housing_features,
    get_feature_columns as housing_feature_cols,
)
from src.models.fraud_autoencoder import FraudAutoencoder
from src.models.lstm_forecaster import LSTMForecaster

# ---------------------------------------------------------------------------
# Raw test DataFrames (small, deterministic)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def credit_risk_df():
    """100-row credit risk DataFrame matching data generator schema."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(18, 75, n),
        "annual_income": rng.uniform(20000, 200000, n).round(2),
        "credit_score": rng.integers(300, 850, n),
        "num_open_accounts": rng.integers(0, 15, n),
        "payment_history_pct": rng.uniform(50, 100, n).round(1),
        "debt_to_income_ratio": rng.uniform(0.05, 0.8, n).round(3),
        "employment_years": rng.uniform(0, 35, n).round(1),
        "loan_amount": rng.uniform(1000, 100000, n).round(2),
        "is_default": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture(scope="session")
def fraud_df():
    """200-row fraud DataFrame matching data generator schema."""
    rng = np.random.default_rng(42)
    n = 200
    merchants = [
        "grocery", "gas_station", "online_retail", "restaurant",
        "electronics", "clothing", "travel", "entertainment",
    ]
    return pd.DataFrame({
        "transaction_amount": rng.uniform(1, 5000, n).round(2),
        "merchant_category": rng.choice(merchants, n),
        "hour_of_day": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "distance_from_home": rng.uniform(0, 200, n).round(1),
        "is_online": rng.integers(0, 2, n),
        "card_age_days": rng.integers(30, 3650, n),
        "num_transactions_last_hour": rng.integers(0, 10, n),
        "amount_vs_avg_ratio": rng.uniform(0.1, 15, n).round(2),
        "is_fraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
    })


@pytest.fixture(scope="session")
def housing_df():
    """100-row housing DataFrame matching data generator schema."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "square_feet": rng.integers(600, 5000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 5, n),
        "year_built": rng.integers(1950, 2024, n),
        "lot_size_sqft": rng.integers(2000, 50000, n),
        "garage_spaces": rng.integers(0, 4, n),
        "has_pool": rng.integers(0, 2, n),
        "neighborhood_tier": rng.integers(1, 6, n),
        "proximity_to_city_center": rng.uniform(1, 50, n).round(1),
        "price": rng.uniform(100000, 1000000, n).round(2),
    })


@pytest.fixture(scope="session")
def timeseries_df():
    """~365-day timeseries DataFrame matching data generator schema."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    products = ["electronics", "clothing", "grocery", "furniture", "sports"]
    rows = []
    rng = np.random.default_rng(42)
    for product in products:
        base = rng.uniform(50, 200)
        for date in dates:
            demand = max(0, base + rng.normal(0, 20))
            rows.append({
                "date": date,
                "product_category": product,
                "demand": round(demand, 1),
                "is_holiday": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tiny trained models (session-scoped — expensive to create, immutable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_credit_risk_model(credit_risk_df):
    """XGBClassifier trained on 100 rows, 5 estimators."""
    df = credit_features(credit_risk_df.copy())
    X = df[credit_feature_cols()]
    y = df["is_default"]
    model = xgb.XGBClassifier(
        n_estimators=5, max_depth=3, learning_rate=0.3,
        objective="binary:logistic", eval_metric="auc",
        random_state=42, verbosity=0,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def tiny_fraud_artifacts(fraud_df):
    """Trained autoencoder, scaler, and isolation forest on 200 rows."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    df, artifacts = fraud_engineer(fraud_df.copy())
    feature_cols = fraud_feature_cols()
    X = df[feature_cols].values.astype(np.float32)
    y = df["is_fraud"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    X_normal = X_scaled[y == 0]

    input_dim = X_scaled.shape[1]
    autoencoder = FraudAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16])
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    X_tensor = torch.FloatTensor(X_normal)
    autoencoder.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(autoencoder(X_tensor), X_tensor)
        loss.backward()
        optimizer.step()
    autoencoder.eval()

    with torch.no_grad():
        errors = autoencoder.reconstruction_error(torch.FloatTensor(X_scaled)).numpy()
    threshold = float(np.percentile(errors, 95))

    iso = IsolationForest(contamination=0.05, n_estimators=10, random_state=42)
    iso.fit(X_scaled)

    return {
        "autoencoder": autoencoder,
        "scaler": scaler,
        "isolation_forest": iso,
        "threshold": threshold,
        "input_dim": input_dim,
    }


@pytest.fixture(scope="session")
def tiny_price_model(housing_df):
    """LGBMRegressor trained on 100 rows, 5 estimators."""
    df = housing_features(housing_df.copy())
    X = df[housing_feature_cols()]
    y = df["price"]
    model = lgb.LGBMRegressor(
        n_estimators=5, max_depth=3, learning_rate=0.3,
        verbose=-1, random_state=42,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def tiny_lstm_artifacts(timeseries_df):
    """Trained LSTM + scalers on tiny timeseries data."""
    from sklearn.preprocessing import MinMaxScaler

    from src.features.timeseries_features import create_sequences

    scalers = {}
    all_X, all_y = [], []

    for product in timeseries_df["product_category"].unique():
        product_df = timeseries_df[timeseries_df["product_category"] == product].sort_values("date")
        values = product_df["demand"].values.astype(float).reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values).flatten()
        scalers[product] = scaler

        X, y = create_sequences(scaled, window_size=30, forecast_horizon=7)
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)

    model = LSTMForecaster(
        input_size=1, hidden_size=16, num_layers=1, dropout=0.0, forecast_horizon=7,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    X_tensor = torch.FloatTensor(X_all[:50])
    y_tensor = torch.FloatTensor(y_all[:50])

    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()

    return {"model": model, "scalers": scalers}


# ---------------------------------------------------------------------------
# Checkpoint directory with saved tiny models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory, tiny_credit_risk_model, tiny_fraud_artifacts,
                   tiny_price_model, tiny_lstm_artifacts):
    """Temporary checkpoint directory with all 4 models saved."""
    base = tmp_path_factory.mktemp("checkpoints")

    # Credit risk
    cr_dir = base / "credit_risk"
    cr_dir.mkdir()
    tiny_credit_risk_model.save_model(str(cr_dir / "model.json"))
    with open(cr_dir / "metadata.json", "w") as f:
        json.dump({
            "problem": "credit_risk", "model_type": "xgboost",
            "metrics": {"test_auc_roc": 0.85}, "timestamp": "2024-01-01T00:00:00",
        }, f)

    # Fraud detection
    fd_dir = base / "fraud_detection"
    fd_dir.mkdir()
    torch.save(tiny_fraud_artifacts["autoencoder"].state_dict(), fd_dir / "autoencoder.pt")
    joblib.dump(tiny_fraud_artifacts["scaler"], fd_dir / "scaler.pkl")
    joblib.dump(tiny_fraud_artifacts["isolation_forest"], fd_dir / "isolation_forest.pkl")
    with open(fd_dir / "metadata.json", "w") as f:
        json.dump({
            "problem": "fraud_detection", "model_type": "autoencoder",
            "metrics": {"anomaly_threshold": tiny_fraud_artifacts["threshold"]},
            "timestamp": "2024-01-01T00:00:00",
        }, f)

    # Price prediction
    pp_dir = base / "price_prediction"
    pp_dir.mkdir()
    joblib.dump(tiny_price_model, pp_dir / "model.pkl")
    with open(pp_dir / "metadata.json", "w") as f:
        json.dump({
            "problem": "price_prediction", "model_type": "lightgbm",
            "metrics": {"test_r2": 0.75}, "timestamp": "2024-01-01T00:00:00",
        }, f)

    # Demand forecasting
    df_dir = base / "demand_forecasting"
    df_dir.mkdir()
    torch.save(tiny_lstm_artifacts["model"].state_dict(), df_dir / "lstm.pt")
    joblib.dump(tiny_lstm_artifacts["scalers"], df_dir / "scalers.pkl")
    with open(df_dir / "metadata.json", "w") as f:
        json.dump({
            "problem": "demand_forecasting", "model_type": "lstm",
            "metrics": {"test_mae": 15.0}, "timestamp": "2024-01-01T00:00:00",
        }, f)

    return base


@pytest.fixture(scope="session")
def predictor(checkpoint_dir, timeseries_df):
    """ModelPredictor using tiny checkpoints from tmp directory."""
    from src.serving.predictor import ModelPredictor

    p = ModelPredictor()
    # Force CPU device for test predictors (models loaded to CPU, avoids MPS mismatch)
    p.device = torch.device("cpu")
    import types

    def patched_ensure_loaded(self, problem):
        if problem in self._models:
            return
        checkpoint_d = checkpoint_dir / problem
        metadata_path = checkpoint_d / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No checkpoint found for {problem}.")

        with open(metadata_path) as f:
            metadata = json.load(f)
        self._artifacts[problem] = {"metadata": metadata}

        if problem == "credit_risk":
            model = xgb.XGBClassifier()
            model.load_model(str(checkpoint_d / "model.json"))
            self._models[problem] = model

        elif problem == "fraud_detection":
            scaler = joblib.load(checkpoint_d / "scaler.pkl")
            self._artifacts[problem]["scaler"] = scaler
            input_dim = len(fraud_feature_cols())
            autoencoder = FraudAutoencoder(input_dim=input_dim, hidden_dims=[64, 32, 16])
            autoencoder.load_state_dict(
                torch.load(checkpoint_d / "autoencoder.pt", weights_only=True, map_location="cpu")
            )
            autoencoder.eval()
            self._models[problem] = autoencoder
            iso = joblib.load(checkpoint_d / "isolation_forest.pkl")
            self._artifacts[problem]["isolation_forest"] = iso
            self._artifacts[problem]["threshold"] = metadata["metrics"].get(
                "anomaly_threshold", 0.18
            )

        elif problem == "price_prediction":
            model = joblib.load(checkpoint_d / "model.pkl")
            self._models[problem] = model

        elif problem == "demand_forecasting":
            scalers = joblib.load(checkpoint_d / "scalers.pkl")
            self._artifacts[problem]["scalers"] = scalers
            model = LSTMForecaster(
                input_size=1, hidden_size=16, num_layers=1, dropout=0.0, forecast_horizon=7,
            )
            model.load_state_dict(
                torch.load(checkpoint_d / "lstm.pt", weights_only=True, map_location="cpu")
            )
            model.eval()
            self._models[problem] = model

    p._ensure_loaded = types.MethodType(patched_ensure_loaded, p)

    # Write a tiny CSV for demand prediction fallback path
    raw_dir = checkpoint_dir.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timeseries_df.to_csv(raw_dir / "daily_demand.csv", index=False)
    p.root = checkpoint_dir.parent

    return p


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_client(predictor):
    """FastAPI TestClient with tiny model predictor injected."""
    from fastapi.testclient import TestClient

    import src.serving.api as api_module

    original_predictor = api_module.predictor
    api_module.predictor = predictor
    client = TestClient(api_module.app)
    yield client
    api_module.predictor = original_predictor
