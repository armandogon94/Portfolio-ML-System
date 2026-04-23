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


class RentalListing(BaseModel):
    """Rental listing inputs for the nightly-rate regressor (A.3)."""

    bedrooms: int = 2
    bathrooms: int = 1
    square_feet: int = 900
    property_type: int = 2  # 1=studio, 2=apartment, 3=house, 4=condo
    location_tier: int = 3  # 1-5 (5 most desirable)
    distance_to_downtown_km: float = 5.0
    amenity_score: int = 6
    peer_nightly_rate: float = 150.0


class DentalAppointment(BaseModel):
    """Dental appointment inputs for the patient no-show classifier (A.4)."""

    age: int = 35
    prior_no_shows: int = 1
    days_until_appointment: int = 14
    appointment_hour: int = 10
    distance_km: float = 8.0
    insurance_type: int = 1
    procedure_complexity: int = 2
    prior_appointments: int = 5


class PatientVitals(BaseModel):
    """Cleveland-style vitals for the heart-disease classifier (A.5)."""

    age: int = 55
    sex: int = 1
    chest_pain_type: int = 3
    resting_bp: int = 130
    cholesterol: int = 240
    max_heart_rate: int = 150
    exercise_angina: int = 0
    oldpeak: float = 1.0


class DeliveryRequest(BaseModel):
    """Shipment inputs for the delivery-ETA regressor (A.7)."""

    distance_km: float = 50
    package_weight_kg: float = 2
    traffic_congestion: int = 3
    weather_severity: int = 1
    time_of_day: int = 10
    day_of_week: int = 2
    carrier_priority: int = 2
    origin_destination_tier: int = 2


class BankCustomer(BaseModel):
    """Bank customer features for the churn classifier (A.6)."""

    tenure_months: int = 48
    balance: float = 50000
    num_products: int = 2
    has_credit_card: int = 1
    is_active_member: int = 1
    estimated_salary: float = 75000
    age: int = 40
    geography_tier: int = 2


class H1BApplication(BaseModel):
    """H-1B visa petition inputs for the approval classifier (A.8)."""

    prevailing_wage: float = 120000
    soc_code_level: int = 3
    employer_size_tier: int = 3
    job_level: int = 2
    education_level: int = 2
    experience_years: int = 5
    country_of_citizenship_tier: int = 2
    employer_prior_approval_rate: float = 0.75


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


@app.post("/predict/rental-price")
async def predict_rental_price(listing: RentalListing):
    try:
        return predictor.predict_rental_price(listing.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/h1b-approval")
async def predict_h1b_approval(app_data: H1BApplication):
    try:
        return predictor.predict_h1b_approval(app_data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/rental-price")
async def explain_rental_price(listing: RentalListing):
    try:
        return predictor.explain_rental_price(listing.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/no-show")
async def predict_no_show(appointment: DentalAppointment):
    try:
        return predictor.predict_dental_noshow(appointment.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/no-show")
async def explain_no_show(appointment: DentalAppointment):
    try:
        return predictor.explain_dental_noshow(appointment.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/heart-disease")
async def predict_heart_disease(vitals: PatientVitals):
    try:
        return predictor.predict_heart_disease(vitals.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/heart-disease")
async def explain_heart_disease(vitals: PatientVitals):
    try:
        return predictor.explain_heart_disease(vitals.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/eta")
async def explain_eta(req: DeliveryRequest):
    try:
        return predictor.explain_delivery_eta(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/churn")
async def predict_churn(customer: BankCustomer):
    try:
        return predictor.predict_customer_churn(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/churn")
async def explain_churn(customer: BankCustomer):
    try:
        return predictor.explain_customer_churn(customer.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/h1b-approval")
async def explain_h1b_approval(app_data: H1BApplication):
    try:
        return predictor.explain_h1b_approval(app_data.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    models = {
        model: {"available": (_CHECKPOINT_ROOT / model / "metadata.json").exists()}
        for model in _ALL_MODELS
    }
    return {"status": "ok", "models": models}
