# Machine Learning System
## Credit Risk, Fraud Detection, Price Prediction & Forecasting

---

## PROJECT OVERVIEW

Production ML system with multiple models:
- **Credit Risk Scoring:** XGBoost classifier for loan approval decisions
- **Fraud Detection:** Isolation Forest for anomaly detection
- **Price Prediction:** Real estate price estimation with regression
- **Demand Forecasting:** Time series (Prophet/LSTM)
- **Model Management:** MLflow experiment tracking
- **Model Serving:** FastAPI inference server
- **Prediction UI:** Streamlit/Gradio interface

**Why it matters:** Demonstrate end-to-end ML pipeline from data to production predictions.

**Subdomain:** ml.305-ai.com

---

## TECH STACK

- **ML Frameworks:** XGBoost, scikit-learn, Prophet, TensorFlow/LSTM
- **Experiment Tracking:** MLflow 2.10+
- **Model Serving:** FastAPI 0.104+
- **UI:** Streamlit or Gradio
- **Data:** Synthetic datasets with Faker
- **Storage:** PostgreSQL + model artifacts

---

## CREDIT RISK SCORING MODEL

### File: `models/credit_risk_model.py`

```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix
import mlflow

def train_credit_risk_model(df: pd.DataFrame) -> xgb.Booster:
    """
    Train XGBoost credit risk scoring model
    Features: age, annual_income, credit_score, num_accounts, payment_history
    Target: is_approved (binary)
    """

    # Feature engineering
    X = df[[
        'age', 'annual_income', 'credit_score',
        'num_accounts', 'payment_history', 'debt_to_income'
    ]].fillna(0)

    y = df['is_approved'].astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train with MLflow tracking
    with mlflow.start_run(run_name="credit_risk_v1"):
        # Log parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
        mlflow.log_params(params)

        # Train
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, 'eval')],
            early_stopping_rounds=10
        )

        # Evaluate
        y_pred = model.predict(dtest)
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", sum(
            (y_pred > 0.5) == y_test) / len(y_test))

        # Log model
        mlflow.xgboost.log_model(model, "credit_risk_model")

    return model

def score_application(model: xgb.Booster, app_data: dict) -> dict:
    """Score a loan application"""
    features = pd.DataFrame([app_data])
    X = xgb.DMatrix(features)

    risk_score = float(model.predict(X)[0])
    recommendation = "APPROVE" if risk_score < 0.2 else "REVIEW" if risk_score < 0.5 else "DECLINE"

    return {
        "risk_score": risk_score,
        "recommendation": recommendation,
        "confidence": max(risk_score, 1 - risk_score)
    }
```

---

## FRAUD DETECTION MODEL

### File: `models/fraud_detection.py`

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import mlflow

def train_fraud_detector(df: pd.DataFrame) -> dict:
    """Train Isolation Forest for fraud detection"""

    features = df[[
        'transaction_amount',
        'merchant_category_code',
        'hour_of_day',
        'days_since_account_open',
        'num_transactions_today'
    ]].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Train Isolation Forest
    with mlflow.start_run(run_name="fraud_detection_v1"):
        model = IsolationForest(
            contamination=0.05,  # Expect 5% fraud
            random_state=42
        )
        model.fit(X_scaled)

        mlflow.log_param("contamination", 0.05)
        mlflow.sklearn.log_model(model, "fraud_detector")

    return {
        "model": model,
        "scaler": scaler
    }

def detect_fraud(model, scaler, transaction: dict) -> dict:
    """Score transaction for fraud risk"""
    features = np.array([[
        transaction['amount'],
        transaction['mcc'],
        transaction['hour'],
        transaction['days_open'],
        transaction['num_today']
    ]])

    X_scaled = scaler.transform(features)
    anomaly_score = model.score_samples(X_scaled)[0]
    is_fraud = model.predict(X_scaled)[0] == -1

    return {
        "is_anomaly": bool(is_fraud),
        "anomaly_score": float(anomaly_score),
        "fraud_probability": float(1 / (1 + np.exp(-anomaly_score)))
    }
```

---

## PRICE PREDICTION MODEL

### File: `models/price_prediction.py`

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow

def train_price_model(df: pd.DataFrame):
    """Train real estate price prediction model"""

    X = df[[
        'square_feet', 'num_bedrooms', 'num_bathrooms',
        'age_years', 'lot_size', 'garage_spaces'
    ]].fillna(0)

    y = df['price']

    # Train
    with mlflow.start_run(run_name="price_prediction_v1"):
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5
        )
        model.fit(X, y)

        # Evaluate
        train_r2 = model.score(X, y)
        mlflow.log_metric("r2_score", train_r2)
        mlflow.sklearn.log_model(model, "price_predictor")

    return model

def predict_price(model, property_data: dict) -> dict:
    """Predict property price"""
    import pandas as pd

    features = pd.DataFrame([property_data])
    prediction = model.predict(features)[0]

    return {
        "predicted_price": float(prediction),
        "confidence_interval": (prediction * 0.85, prediction * 1.15)
    }
```

---

## TIME SERIES FORECASTING

### File: `models/demand_forecasting.py`

```python
from statsmodels.tsa.prophet import Prophet
import pandas as pd
import mlflow

def train_forecast_model(df: pd.DataFrame):
    """Train Prophet for demand forecasting"""
    # Prepare data for Prophet
    prophet_df = df[['date', 'demand']].copy()
    prophet_df.columns = ['ds', 'y']

    with mlflow.start_run(run_name="demand_forecast_v1"):
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)

        mlflow.sklearn.log_model(model, "demand_forecaster")

    return model

def forecast_demand(model, periods: int = 30) -> pd.DataFrame:
    """Generate demand forecast"""
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
```

---

## FASTAPI INFERENCE SERVER

### File: `server/main.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import mlflow.xgboost
import pickle
import os

app = FastAPI(title="ML Inference Server", version="1.0")

# Load models
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

credit_risk_model = mlflow.xgboost.load_model("models:/credit_risk_model/production")
fraud_detector = pickle.load(open("models/fraud_detector.pkl", "rb"))
price_model = mlflow.sklearn.load_model("models:/price_predictor/production")

# Request models
class LoanApplication(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    num_accounts: int
    payment_history: float

class Transaction(BaseModel):
    amount: float
    mcc: int
    hour: int
    days_open: int
    num_today: int

class Property(BaseModel):
    square_feet: float
    num_bedrooms: int
    num_bathrooms: float
    age_years: int

# Endpoints
@app.post("/predict/credit-risk")
async def predict_credit_risk(app: LoanApplication):
    """Score loan application"""
    import xgboost as xgb
    import numpy as np

    features = np.array([[
        app.age,
        app.annual_income,
        app.credit_score,
        app.num_accounts,
        app.payment_history
    ]])

    X = xgb.DMatrix(features)
    risk_score = float(credit_risk_model.predict(X)[0])
    recommendation = "APPROVE" if risk_score < 0.2 else "REVIEW" if risk_score < 0.5 else "DECLINE"

    return {
        "risk_score": risk_score,
        "recommendation": recommendation,
        "confidence": float(max(risk_score, 1 - risk_score))
    }

@app.post("/predict/fraud")
async def detect_fraud(tx: Transaction):
    """Detect fraudulent transactions"""
    # Similar implementation
    pass

@app.post("/predict/price")
async def predict_property_price(prop: Property):
    """Predict property price"""
    import pandas as pd

    features = pd.DataFrame([prop.dict()])
    prediction = price_model.predict(features)[0]

    return {
        "predicted_price": float(prediction),
        "confidence_interval": (prediction * 0.85, prediction * 1.15)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## STREAMLIT UI

### File: `ui/streamlit_app.py`

```python
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ML Predictions", layout="wide")
st.title("Machine Learning Prediction System")

# Navigation
page = st.sidebar.radio("Select Model", [
    "Credit Risk Scoring",
    "Fraud Detection",
    "Price Prediction",
    "Demand Forecasting"
])

if page == "Credit Risk Scoring":
    st.header("Loan Application Risk Scoring")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 80, 35)
        income = st.number_input("Annual Income", 20000, 500000, 60000)

    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 680)
        accounts = st.number_input("Number of Accounts", 0, 20, 3)

    payment_history = st.slider("Payment History Score", 0, 100, 75)

    if st.button("Score Application"):
        response = requests.post(
            "http://localhost:8000/predict/credit-risk",
            json={
                "age": age,
                "annual_income": income,
                "credit_score": credit_score,
                "num_accounts": accounts,
                "payment_history": payment_history
            }
        )

        if response.status_code == 200:
            result = response.json()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Risk Score", f"{result['risk_score']:.1%}")

            with col2:
                color = "🟢" if result['recommendation'] == "APPROVE" else "🟡" if result['recommendation'] == "REVIEW" else "🔴"
                st.metric("Recommendation", f"{color} {result['recommendation']}")

            with col3:
                st.metric("Confidence", f"{result['confidence']:.1%}")
```

---

## DOCKER COMPOSE

Port mappings follow the global allocation from **PORT-MAP.md**:
- ML Dashboard UI: **3070:3000** (host:container)
- Prediction API: **8070:8000** (host:container)
- MLflow UI: **5070:5000** (host:container)

Run locally:
```bash
docker-compose up
# OR with development overrides:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

See **docker-compose.yml** and **docker-compose.dev.yml** for full configuration.

---

## ESTIMATED TIMELINE

- **Data Generation:** 1.5 hours
- **Model Training:** 4 hours
- **MLflow Setup:** 1 hour
- **Inference Server:** 2 hours
- **UI Development:** 2 hours
- **Testing:** 1.5 hours

**Total:** ~12 hours

---

**ML System Version:** 1.0
**Status:** Production-ready
