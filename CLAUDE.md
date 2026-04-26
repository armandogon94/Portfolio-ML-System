# Portfolio ML System

> **Port allocation:** See [PORTS.md](PORTS.md) before changing any docker-compose ports. All ports outside the assigned ranges are taken by other projects.

Production ML system with 10 models across 6 industries (Real Estate, Dental, Healthcare, Fintech, Logistics, Legal/Immigration). Backed by FastAPI + a Next.js 14 demo UI with shadcn/ui. Live model dashboard at `/dashboard` summarizes status + key metrics for the full 20-model catalog.

## Tech Stack

Python 3.11+, uv, PyTorch (MPS backend), XGBoost, LightGBM, scikit-learn, W&B, MLflow, FastAPI, Next.js 14 (App Router), TypeScript, TailwindCSS, shadcn/ui, Recharts, Zod, TanStack Query

## Commands

```bash
make setup       # Install Python dependencies (uv sync --extra dev)
make data        # Generate all synthetic datasets
make train       # Train all models
make evaluate    # Evaluate all models, save CSV results
make serve       # Launch FastAPI inference server (port 8070)
make test        # Run pytest (excludes network + parity by default)
make lint        # Run ruff linter
make all         # Full pipeline: data + train + evaluate
make web-dev     # Launch Next.js demo UI at http://localhost:3071
make docker-up   # Bring up the 3-service stack (mlflow, ml-api, ml-web)
```

Individual scripts:
```bash
uv run python scripts/generate_data.py --problem credit_risk|fraud|housing|timeseries|all
uv run python scripts/train.py --model credit_risk|fraud|price|forecaster|all \
    [--modality synthetic|stream|mixed|all] [--no-wandb]
uv run python scripts/evaluate.py --model credit_risk|fraud|price|forecaster|all
```

**Streaming datasets (Phase A.1+):** Models whose config declares `data.kaggle_slug`
can train on real Kaggle data without committing it to the repo. Three modalities:
- `--modality synthetic` — existing generator path (default, backward-compatible)
- `--modality stream` — real Kaggle dataset streamed via `kagglehub`, adapted to canonical schema
- `--modality mixed` — synthetic + real concatenated (with `modality` column as feature)
- `--modality all` — trains all three under one MLflow parent run, writes a comparison
  CSV, flags the best-metric variant as "recommended" in its `metadata.json`

Kaggle creds go in `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `~/.kaggle/kaggle.json`.
Real data caches to `~/.cache/kagglehub/` — **never** in `data/raw/`.

## Architecture

- `configs/` - YAML configs with all hyperparameters (never hardcoded)
- `src/config.py` - YAML config loader with path resolution
- `src/device.py` - MPS/CUDA/CPU auto-detection for PyTorch
- `src/data/` - Synthetic data generators (Faker + numpy)
- `src/features/` - Feature engineering pipelines per problem
- `src/models/` - Model definitions (XGBoost, LightGBM, PyTorch Autoencoder, PyTorch LSTM)
- `src/training/` - BaseTrainer + problem-specific trainers with W&B integration
- `src/evaluation/` - Metric computation, CSV export
- `src/serving/` - ModelPredictor (checkpoint loading + inference), FastAPI server
- `scripts/` - CLI entry points for generate, train, evaluate, serve
- `web/` - Next.js 14 demo UI (per-industry pages + /dashboard)
- `checkpoints/` - Model weights + metadata.json (gitignored)
- `results/` - CSV evaluation results (committed)
- `docs/decisions/` - ADRs (e.g., ADR-001 records the Gradio → Next.js migration)

## Conventions

- All scripts run via CLI (no notebooks)
- YAML configs drive all hyperparameters
- Every training run saves: checkpoint + metadata.json + results CSV
- W&B logging is optional (local JSON fallback always writes)
- PyTorch models use MPS on Apple Silicon (`src/device.py`)
- Synthetic data makes project self-contained; real datasets stream from Kaggle/HF and cache at `~/.cache/kagglehub/` — never in `data/raw/`
- Network-backed tests live under `@pytest.mark.network` and are skipped by default in `make test`; run them explicitly with `uv run pytest -m network`

## Deep Learning Models

- **Fraud Autoencoder** (`src/models/fraud_autoencoder.py`): Input->64->32->16->32->64->Output. Trained on normal transactions only. Anomaly = high reconstruction error. Runs on MPS.
- **LSTM Forecaster** (`src/models/lstm_forecaster.py`): LSTM(hidden=64, layers=2, dropout=0.2). 30-day sliding window input, 7-day forecast output. Runs on MPS.

## Testing

```bash
uv run pytest tests/ -v
```

Tests cover data generation, feature engineering, model inference, and serving.
