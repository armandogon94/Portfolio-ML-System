# Portfolio ML System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production ML system demonstrating end-to-end machine learning engineering: synthetic data generation, feature engineering, model training with experiment tracking, model checkpointing, and a unified web prediction interface. Built to run on **Apple Silicon** with PyTorch MPS acceleration.

---

## Models

| Problem | Algorithm | Type | Device | Key Metric | Score |
|---------|-----------|------|--------|------------|-------|
| Credit Risk Scoring | XGBoost Classifier | Classification | CPU | AUC-ROC | 0.888 |
| Fraud Detection | PyTorch Autoencoder + Isolation Forest | Anomaly Detection | **MPS** | AUC-ROC | 0.964 |
| Real Estate Pricing | LightGBM Regressor | Regression | CPU | R2 | 0.942 |
| Demand Forecasting | PyTorch LSTM | Time Series | **MPS** | Avg MAE | 22.2 |

---

## Architecture

```
Data Generation     Feature Engineering     Training + Tracking     Serving                  Demo UI
 (Faker + NumPy)     (scikit-learn)          (W&B + MLflow)         (FastAPI)               (Next.js 14)

  credit_risk ──┐                                                                  ┌──> /fintech/credit-risk
  fraud ────────│                                                                  │    /fintech/fraud
  price ────────├──> feature pipelines ──> XGBoost / LightGBM ──> checkpoints ──> ML├──> /fintech/churn
  rental_price ─│                          / Autoencoder (MPS) /                  api │    /real-estate/price
  no-show ──────│                          / LSTM (MPS)         /                 :8070    /real-estate/rental-price
  ...           │                                                                  │    /dental/no-show
                                                                                   │    /healthcare/heart-disease
                                                                                   │    /logistics/eta
                                                                                   │    /logistics/demand
                                                                                   │    /legal/h1b-approval
                                                                                   └──> /dashboard  (live status of all 20)
```

---

## Quick Start

**Prerequisites:** Python 3.11+, macOS/Linux

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/YOUR_USERNAME/portfolio-ml-system.git
cd portfolio-ml-system
uv sync

# Run the full pipeline
make all          # Generate data -> Train models -> Evaluate

# Launch the Next.js demo UI (native, fastest)
make web-dev      # Opens at http://localhost:3071

# Or run the full Docker stack — 3 services (mlflow, ml-api, ml-web)
make docker-up    # /dashboard at http://localhost:3071/dashboard
```

Or step-by-step:

```bash
uv run python scripts/generate_data.py --problem all
uv run python scripts/train.py --model all --no-wandb
uv run python scripts/evaluate.py
uv run python scripts/serve.py           # FastAPI inference server (port 8070)
cd web && pnpm install && pnpm dev       # Next.js demo UI (port 3071)
```

---

## Streaming Datasets (Phase A.1+)

Models whose config declares `data.kaggle_slug` can be trained on **real public datasets** without ever committing them to the repo. Each such model supports three training modalities:

| Modality | Data source | Use case |
|---|---|---|
| `synthetic` | Bundled generator | Self-contained demo (no network, default) |
| `stream` | Real Kaggle dataset, streamed via `kagglehub` | Best-case "real data" performance |
| `mixed` | Concatenation of both, with a `modality` feature column | Demonstrates augmentation value |

Train all three and let the CLI pick a winner:

```bash
uv run python scripts/train.py --model price --modality all
```

This creates one MLflow parent run with three nested children, writes three modality-specific checkpoints under `checkpoints/price_prediction_<modality>/`, and emits `results/modality_comparison_price_prediction.csv` with the ranked comparison. The best-metric variant is flagged `recommended: true` in its `metadata.json` — that's the checkpoint the demo UI should load.

**Kaggle credentials** — one of:

```bash
# Option 1: environment variables (copy .env.example → .env, fill in)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option 2: standard Kaggle location
# Drop kaggle.json at ~/.kaggle/kaggle.json
```

Get an API key at [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → *Create New Token*.

**Cache location:** Real datasets are cached to `~/.cache/kagglehub/` (outside the repo, outside Docker build context). `data/raw/` stays small even after training on all three modalities.

**Running network tests:** The mocked test suite catches logic bugs; a single `@pytest.mark.network` test in `tests/test_streaming.py` is the canary for real-Kaggle integration. It's skipped by default; run it explicitly once creds are set:

```bash
uv run pytest -m network tests/test_streaming.py
```

---

## Project Structure

```
portfolio-ml-system/
├── configs/                    # YAML configs (hyperparameters, data paths)
│   ├── credit_risk.yaml
│   ├── fraud_detection.yaml
│   ├── price_prediction.yaml
│   └── demand_forecasting.yaml
├── src/
│   ├── config.py               # Config loader
│   ├── device.py               # MPS/CUDA/CPU auto-detection
│   ├── data/                   # Synthetic data generators
│   ├── features/               # Feature engineering pipelines
│   ├── models/                 # Model architectures
│   ├── training/               # Trainers with W&B integration
│   ├── evaluation/             # Metric computation
│   └── serving/                # Predictor + FastAPI server
├── scripts/                    # CLI entry points
├── web/                        # Next.js 14 demo UI (per-industry pages + /dashboard)
├── docs/decisions/             # ADRs (e.g., ADR-001 = Gradio → Next.js migration)
├── checkpoints/                # Saved model weights + metadata
├── results/                    # Evaluation CSVs
└── tests/                      # pytest test suite
```

---

## Training Pipeline

All hyperparameters live in YAML config files -- nothing is hardcoded in training scripts.

```bash
# Train a specific model
uv run python scripts/train.py --model credit_risk
uv run python scripts/train.py --model fraud
uv run python scripts/train.py --model price
uv run python scripts/train.py --model forecaster

# Train all models
uv run python scripts/train.py --model all

# Disable W&B (use local logging only)
uv run python scripts/train.py --model all --no-wandb
```

Each training run produces:
- `checkpoints/<model>/model.*` -- model weights
- `checkpoints/<model>/metadata.json` -- metrics, hyperparams, timestamps
- `results/<model>_metrics.csv` -- evaluation results

---

## Apple Silicon MPS

Two of the four models use PyTorch with the Metal Performance Shaders (MPS) backend for GPU acceleration on Apple Silicon:

- **Fraud Autoencoder**: Symmetric autoencoder (Input->64->32->16->32->64->Output) trained on normal transactions. Anomaly = high reconstruction error.
- **LSTM Forecaster**: 2-layer LSTM with 64 hidden units, 30-day lookback, 7-day forecast.

Device is auto-detected via `src/device.py`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for operations not yet supported on MPS (handled automatically).

---

## Experiment Tracking

**Weights & Biases** (free tier) is supported for cloud experiment tracking:

```bash
# Set your W&B API key
export WANDB_API_KEY=your_key_here

# Train with W&B logging
uv run python scripts/train.py --model all
```

Without an API key, all metrics are still saved locally to `checkpoints/*/metadata.json` and `results/*.csv`.

---

## Web Interface

The Next.js demo UI (`web/`) hosts one page per ready model plus a live dashboard:

- **Landing** (`/`) — 6 industry tiles linking into their model lists
- **/dashboard** — sortable table of all 20 cataloged models with status badges, key metrics, and MLflow training-history sparklines for the ready ones; auto-refreshes every 30 s via Next.js ISR
- **Per-industry model pages** — one page per ready model, each with a Zod-validated input form on the left and the model's prediction + SHAP/gradient explanation on the right:
  - `/fintech/credit-risk` · `/fintech/fraud` · `/fintech/churn`
  - `/real-estate/price` · `/real-estate/rental-price`
  - `/dental/no-show`
  - `/healthcare/heart-disease`
  - `/logistics/eta` · `/logistics/demand`
  - `/legal/h1b-approval`

```bash
make web-dev      # Native pnpm dev — fastest iteration, hot reload
make docker-up    # Full Docker stack (3 services), demo at http://localhost:3071
```

The original Gradio prototype was retired in Phase A.9 — see `docs/decisions/ADR-001-gradio-to-nextjs.md` for the rationale.

---

## API Server

FastAPI inference server with REST endpoints:

```bash
make serve  # Launches at http://localhost:8000

# Endpoints
curl -X POST http://localhost:8000/predict/credit-risk -H "Content-Type: application/json" \
  -d '{"age": 35, "annual_income": 65000, "credit_score": 700, "num_open_accounts": 3, "payment_history_pct": 85, "debt_to_income_ratio": 0.3, "employment_years": 8, "loan_amount": 25000}'

curl -X POST http://localhost:8000/predict/fraud -H "Content-Type: application/json" \
  -d '{"transaction_amount": 500, "merchant_category": "electronics", "hour_of_day": 2, "day_of_week": 3, "distance_from_home": 100, "is_online": 1, "card_age_days": 30, "num_transactions_last_hour": 5, "amount_vs_avg_ratio": 10}'

curl http://localhost:8000/health
curl http://localhost:8000/models
```

---

## Testing

```bash
make test                    # Run all tests
uv run pytest tests/ -v      # Verbose output
```

---

## Data

All datasets are **synthetic**, generated via Python scripts with realistic statistical distributions. No external downloads required -- the project is fully self-contained.

| Dataset | Rows | Features | Generation Method |
|---------|------|----------|------------------|
| Credit Risk | 50,000 | 8 + engineered | Log-normal income, logistic default target |
| Fraud | 200,000 | 9 + engineered | Shifted distributions for fraud patterns |
| Housing | 30,000 | 9 + engineered | Polynomial price function + noise |
| Time Series | 5,475 | 4 | Trend + seasonality + holidays + noise |

---

## Tech Stack

- **ML**: PyTorch, XGBoost, LightGBM, scikit-learn
- **Tracking**: Weights & Biases (free tier) + MLflow (always-on local)
- **Web UI**: Next.js 14 (App Router) + TypeScript + TailwindCSS + shadcn/ui + Recharts
- **API**: FastAPI + Uvicorn
- **Data**: Faker, NumPy, Pandas; real Kaggle datasets streamed via `kagglehub`
- **Environment**: uv (10-100x faster than pip), pnpm for the web app
- **Hardware**: Apple Silicon MPS acceleration

---

## License

MIT
