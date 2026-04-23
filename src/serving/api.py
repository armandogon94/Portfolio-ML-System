"""FastAPI inference server."""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.logging_config import setup_logging
from src.serving.predictor import ModelPredictor

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Inference Server", version="1.0.0")
predictor = ModelPredictor()

_CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent.parent / "checkpoints"
_ALL_MODELS = [
    "credit_risk",
    "fraud_detection",
    "price_prediction",
    "demand_forecasting",
    "delivery_eta",
]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status code, and latency for every request (excluding /health)."""
    if request.url.path == "/health":
        return await call_next(request)

    start = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start) * 1000, 1)

    msg = "%s %s %d %.1fms" % (
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    if response.status_code >= 500:
        logger.error(msg)
    elif response.status_code >= 400:
        logger.warning(msg)
    else:
        logger.info(msg)

    return response


class LoanApplication(BaseModel):
    age: int = 35
    annual_income: float = 65000
    credit_score: int = 700
    num_open_accounts: int = 3
    payment_history_pct: float = 85.0
    debt_to_income_ratio: float = 0.3
    employment_years: float = 8.0
    loan_amount: float = 25000


class Transaction(BaseModel):
    transaction_amount: float = 150.0
    merchant_category: str = "online_retail"
    hour_of_day: int = 14
    day_of_week: int = 2
    distance_from_home: float = 15.0
    is_online: int = 1
    card_age_days: int = 365
    num_transactions_last_hour: int = 1
    amount_vs_avg_ratio: float = 3.0


class Property(BaseModel):
    square_feet: int = 1800
    bedrooms: int = 3
    bathrooms: int = 2
    year_built: int = 2000
    lot_size_sqft: int = 8000
    garage_spaces: int = 2
    has_pool: int = 0
    neighborhood_tier: int = 3
    proximity_to_city_center: float = 10.0


class DemandRequest(BaseModel):
    product: str = "electronics"


class DeliveryRequest(BaseModel):
    distance_km: float = 50
    package_weight_kg: float = 2
    traffic_congestion: int = 3
    weather_severity: int = 1
    time_of_day: int = 10
    day_of_week: int = 2
    carrier_priority: int = 2
    origin_destination_tier: int = 2


@app.post("/predict/credit-risk")
async def predict_credit_risk(app_data: LoanApplication):
    try:
        return predictor.predict_credit_risk(app_data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/fraud")
async def predict_fraud(tx: Transaction):
    try:
        return predictor.predict_fraud(tx.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/price")
async def predict_price(prop: Property):
    try:
        return predictor.predict_price(prop.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/demand")
async def predict_demand(req: DemandRequest):
    try:
        return predictor.predict_demand(req.product)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/eta")
async def predict_eta(req: DeliveryRequest):
    try:
        return predictor.predict_delivery_eta(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def get_models():
    return predictor.get_model_info()


@app.post("/explain/credit-risk")
async def explain_credit_risk(app_data: LoanApplication):
    try:
        return predictor.explain_credit_risk(app_data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/price")
async def explain_price(prop: Property):
    try:
        return predictor.explain_price(prop.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/fraud")
async def explain_fraud(tx: Transaction):
    try:
        return predictor.explain_fraud(tx.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/eta")
async def explain_eta(req: DeliveryRequest):
    try:
        return predictor.explain_delivery_eta(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    models = {
        model: {"available": (_CHECKPOINT_ROOT / model / "metadata.json").exists()}
        for model in _ALL_MODELS
    }
    return {"status": "ok", "models": models}
