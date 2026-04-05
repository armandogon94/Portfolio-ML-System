# Portfolio ML System

Production ML system with 4 models: Credit Risk (XGBoost), Fraud Detection (PyTorch Autoencoder), Price Prediction (LightGBM), Demand Forecasting (PyTorch LSTM). Unified Gradio web interface.

## Tech Stack

Python 3.11+, uv, PyTorch (MPS backend), XGBoost, LightGBM, scikit-learn, W&B, Gradio, FastAPI

## Commands

```bash
make setup      # Install dependencies (uv sync --extra dev)
make data       # Generate all 4 synthetic datasets
make train      # Train all 4 models
make evaluate   # Evaluate all models, save CSV results
make ui         # Launch Gradio web interface
make serve      # Launch FastAPI inference server
make test       # Run pytest
make lint       # Run ruff linter
make all        # Full pipeline: data + train + evaluate
```

Individual scripts:
```bash
uv run python scripts/generate_data.py --problem credit_risk|fraud|housing|timeseries|all
uv run python scripts/train.py --model credit_risk|fraud|price|forecaster|all [--no-wandb]
uv run python scripts/evaluate.py --model credit_risk|fraud|price|forecaster|all
```

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
- `app/gradio_app.py` - Unified Gradio UI with 5 tabs
- `checkpoints/` - Model weights + metadata.json (gitignored)
- `results/` - CSV evaluation results (committed)

## Conventions

- All scripts run via CLI (no notebooks)
- YAML configs drive all hyperparameters
- Every training run saves: checkpoint + metadata.json + results CSV
- W&B logging is optional (local JSON fallback always writes)
- PyTorch models use MPS on Apple Silicon (`src/device.py`)
- Synthetic data makes project self-contained (no external downloads)

## Deep Learning Models

- **Fraud Autoencoder** (`src/models/fraud_autoencoder.py`): Input->64->32->16->32->64->Output. Trained on normal transactions only. Anomaly = high reconstruction error. Runs on MPS.
- **LSTM Forecaster** (`src/models/lstm_forecaster.py`): LSTM(hidden=64, layers=2, dropout=0.2). 30-day sliding window input, 7-day forecast output. Runs on MPS.

## Testing

```bash
uv run pytest tests/ -v
```

Tests cover data generation, feature engineering, model inference, and serving.
