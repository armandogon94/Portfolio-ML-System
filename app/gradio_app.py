"""Unified Gradio web interface with 5 tabs for all ML models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.serving.predictor import ModelPredictor

predictor = ModelPredictor()


# ── Tab 1: Credit Risk ──────────────────────────────────────────────────────

def predict_credit_risk(age, income, credit_score, accounts, payment_pct, dti, emp_years, loan_amt):
    result = predictor.predict_credit_risk({
        "age": age,
        "annual_income": income,
        "credit_score": credit_score,
        "num_open_accounts": accounts,
        "payment_history_pct": payment_pct,
        "debt_to_income_ratio": dti,
        "employment_years": emp_years,
        "loan_amount": loan_amt,
    })

    risk_pct = f"{result['risk_score']:.1%}"
    confidence_pct = f"{result['confidence']:.1%}"
    rec = result["recommendation"]

    color = {"APPROVE": "green", "REVIEW": "orange", "DECLINE": "red"}.get(rec, "gray")
    summary = (
        f"### Risk Score: {risk_pct}\n"
        f"### Recommendation: <span style='color:{color};font-weight:bold'>{rec}</span>\n"
        f"### Confidence: {confidence_pct}"
    )
    return summary


def build_credit_risk_tab():
    with gr.TabItem("Credit Risk Scoring"):
        gr.Markdown("## Loan Application Risk Assessment\nEnter applicant details to get a credit risk score.")
        with gr.Row():
            with gr.Column():
                age = gr.Slider(18, 75, value=35, step=1, label="Age")
                income = gr.Number(value=65000, label="Annual Income ($)")
                credit_score = gr.Slider(300, 850, value=700, step=1, label="Credit Score")
                accounts = gr.Slider(0, 20, value=3, step=1, label="Open Accounts")
            with gr.Column():
                payment_pct = gr.Slider(0, 100, value=85, step=1, label="Payment History (%)")
                dti = gr.Slider(0, 1.5, value=0.3, step=0.01, label="Debt-to-Income Ratio")
                emp_years = gr.Slider(0, 40, value=8, step=0.5, label="Employment Years")
                loan_amt = gr.Number(value=25000, label="Loan Amount ($)")

        btn = gr.Button("Score Application", variant="primary")
        output = gr.Markdown()
        btn.click(predict_credit_risk, [age, income, credit_score, accounts, payment_pct, dti, emp_years, loan_amt], output)


# ── Tab 2: Fraud Detection ──────────────────────────────────────────────────

MERCHANT_CATS = [
    "grocery", "restaurant", "gas_station", "online_retail", "electronics",
    "clothing", "travel", "entertainment", "healthcare", "utilities",
    "education", "home_improvement", "automotive", "subscription", "atm_withdrawal",
]

def predict_fraud(amount, merchant, hour, day, distance, is_online, card_age, num_tx, ratio):
    result = predictor.predict_fraud({
        "transaction_amount": amount,
        "merchant_category": merchant,
        "hour_of_day": hour,
        "day_of_week": day,
        "distance_from_home": distance,
        "is_online": int(is_online),
        "card_age_days": card_age,
        "num_transactions_last_hour": num_tx,
        "amount_vs_avg_ratio": ratio,
    })

    risk = result["risk_level"]
    color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}.get(risk, "gray")

    summary = (
        f"### Risk Level: <span style='color:{color};font-weight:bold'>{risk}</span>\n"
        f"### Fraud Probability: {result['fraud_probability']:.1%}\n"
        f"**Reconstruction Error:** {result['reconstruction_error']:.6f} "
        f"(threshold: {result['anomaly_threshold']:.6f})\n\n"
        f"**Autoencoder Anomaly:** {'Yes' if result['is_anomaly_autoencoder'] else 'No'} | "
        f"**Isolation Forest Anomaly:** {'Yes' if result['is_anomaly_isolation_forest'] else 'No'}"
    )
    return summary


def build_fraud_tab():
    with gr.TabItem("Fraud Detection"):
        gr.Markdown("## Transaction Fraud Analysis\nEnter transaction details to detect potential fraud.")
        with gr.Row():
            with gr.Column():
                amount = gr.Number(value=150, label="Transaction Amount ($)")
                merchant = gr.Dropdown(MERCHANT_CATS, value="online_retail", label="Merchant Category")
                hour = gr.Slider(0, 23, value=14, step=1, label="Hour of Day")
                day = gr.Slider(0, 6, value=2, step=1, label="Day of Week (0=Mon)")
            with gr.Column():
                distance = gr.Number(value=15, label="Distance from Home (miles)")
                is_online = gr.Checkbox(value=True, label="Online Transaction")
                card_age = gr.Slider(30, 3650, value=365, step=1, label="Card Age (days)")
                num_tx = gr.Slider(0, 20, value=1, step=1, label="Transactions Last Hour")
                ratio = gr.Slider(0, 20, value=3.0, step=0.1, label="Amount vs Avg Ratio")

        btn = gr.Button("Analyze Transaction", variant="primary")
        output = gr.Markdown()
        btn.click(predict_fraud, [amount, merchant, hour, day, distance, is_online, card_age, num_tx, ratio], output)


# ── Tab 3: Price Prediction ─────────────────────────────────────────────────

def predict_price(sqft, beds, baths, year, lot, garage, pool, tier, proximity):
    result = predictor.predict_price({
        "square_feet": sqft,
        "bedrooms": beds,
        "bathrooms": baths,
        "year_built": year,
        "lot_size_sqft": lot,
        "garage_spaces": garage,
        "has_pool": int(pool),
        "neighborhood_tier": tier,
        "proximity_to_city_center": proximity,
    })

    summary = (
        f"### Predicted Price: ${result['predicted_price']:,.0f}\n"
        f"### Price Range: ${result['price_range_low']:,.0f} - ${result['price_range_high']:,.0f}\n"
        f"*(90% confidence interval)*"
    )
    return summary


def build_price_tab():
    with gr.TabItem("Price Prediction"):
        gr.Markdown("## Real Estate Price Estimation\nEnter property details to get a price prediction.")
        with gr.Row():
            with gr.Column():
                sqft = gr.Slider(400, 8000, value=1800, step=50, label="Square Feet")
                beds = gr.Slider(1, 6, value=3, step=1, label="Bedrooms")
                baths = gr.Slider(1, 5, value=2, step=1, label="Bathrooms")
                year = gr.Slider(1950, 2024, value=2000, step=1, label="Year Built")
            with gr.Column():
                lot = gr.Slider(1000, 50000, value=8000, step=500, label="Lot Size (sqft)")
                garage = gr.Slider(0, 3, value=2, step=1, label="Garage Spaces")
                pool = gr.Checkbox(value=False, label="Has Pool")
                tier = gr.Slider(1, 5, value=3, step=1, label="Neighborhood Tier (1=budget, 5=luxury)")
                proximity = gr.Slider(0.5, 50, value=10, step=0.5, label="Distance to City Center (miles)")

        btn = gr.Button("Estimate Price", variant="primary")
        output = gr.Markdown()
        btn.click(predict_price, [sqft, beds, baths, year, lot, garage, pool, tier, proximity], output)


# ── Tab 4: Demand Forecasting ───────────────────────────────────────────────

PRODUCTS = ["electronics", "clothing", "groceries", "furniture", "sports"]

def predict_demand(product):
    result = predictor.predict_demand(product)

    predictions = result["predictions"]
    days = list(range(1, len(predictions) + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Day {d}" for d in days], y=predictions, marker_color="steelblue"))
    fig.update_layout(
        title=f"7-Day Demand Forecast: {product.title()}",
        xaxis_title="Day",
        yaxis_title="Predicted Demand (units)",
        template="plotly_white",
        height=400,
    )

    summary = (
        f"### Average Predicted Demand: {result['avg_predicted_demand']:.0f} units/day\n"
        f"**Product:** {product.title()} | **Horizon:** {result['forecast_days']} days"
    )
    return fig, summary


def build_forecast_tab():
    with gr.TabItem("Demand Forecasting"):
        gr.Markdown("## Product Demand Forecasting\nSelect a product category to see the 7-day demand forecast.")
        product = gr.Dropdown(PRODUCTS, value="electronics", label="Product Category")
        btn = gr.Button("Generate Forecast", variant="primary")

        plot = gr.Plot()
        output = gr.Markdown()
        btn.click(predict_demand, [product], [plot, output])


# ── Tab 5: Dashboard ────────────────────────────────────────────────────────

def load_dashboard():
    info = predictor.get_model_info()
    rows = []
    for problem, meta in info.items():
        metrics = meta.get("metrics", {})
        key_metrics = {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)}
        top = dict(list(key_metrics.items())[:4])
        rows.append({
            "Model": problem.replace("_", " ").title(),
            "Type": meta.get("model_type", "N/A"),
            "Training Time": f"{meta.get('training_time_seconds', 0):.1f}s",
            **top,
        })

    df = pd.DataFrame(rows)

    import torch
    dev = {
        "device": str(predictor.device),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    sys_info = (
        f"**Device:** {dev['device']} | "
        f"**PyTorch:** {dev['pytorch_version']} | "
        f"**MPS Available:** {dev['mps_available']}"
    )
    return df, sys_info


def build_dashboard_tab():
    with gr.TabItem("Model Dashboard"):
        gr.Markdown("## Model Performance Summary")
        btn = gr.Button("Refresh Dashboard", variant="secondary")
        table = gr.Dataframe()
        sys_info = gr.Markdown()
        btn.click(load_dashboard, [], [table, sys_info])


# ── Main App ────────────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(title="Portfolio ML System", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Portfolio ML System\n"
            "Production ML system with credit risk scoring, fraud detection, "
            "price prediction, and demand forecasting.\n"
            "Built with PyTorch (Apple Silicon MPS), XGBoost, LightGBM, and Gradio."
        )
        with gr.Tabs():
            build_credit_risk_tab()
            build_fraud_tab()
            build_price_tab()
            build_forecast_tab()
            build_dashboard_tab()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
