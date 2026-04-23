# Spec: Portfolio ML System — Production Hardening

## Objective

Transform an existing 4-model ML system from a working prototype into a production-grade portfolio piece that demonstrates full MLOps maturity. The core ML pipeline (data generation, feature engineering, training, evaluation, serving) is already solid. What's missing is the production engineering layer: working Docker containers, comprehensive tests, MLflow model registry, model explainability, and structured logging.

**Target audience:** Technical reviewers (hiring managers, senior engineers) evaluating ML engineering depth.

**Success looks like:** A single `docker compose up` starts the entire ML system — API, UI, and MLflow — with all services healthy. Tests cover 80%+ of the codebase. Models are versioned in MLflow. Predictions come with explanations (SHAP values). Logs are structured JSON.

---

## Current State

### What Works

| Component | Status | Key Metric |
|-----------|--------|-----------|
| Credit Risk (XGBoost) | Trained | AUC-ROC: 0.888 |
| Fraud Detection (Autoencoder + IsoForest) | Trained | AUC-ROC: 0.964 |
| Price Prediction (LightGBM) | Trained | R²: 0.942 |
| Demand Forecasting (LSTM) | Trained | MAE: 22.2 |
| Synthetic Data Generators | Working | 4 generators, self-contained |
| Feature Engineering | Working | 4 pipelines, domain-specific |
| BaseTrainer (W&B integration) | Working | Optional W&B, local JSON fallback |
| FastAPI Server | Working | 4 endpoints + health check |
| Gradio UI | Working | 5 tabs (4 models + dashboard) |
| YAML Configs | Working | All hyperparameters externalized |
| Device Auto-Detection | Working | MPS > CUDA > CPU |

### What's Broken or Missing

| Gap | Impact |
|-----|--------|
| Dockerfiles don't exist | docker-compose.yml is non-functional |
| docker-compose references external PostgreSQL/Redis | Can't run standalone |
| Test coverage ~40% | Smoke tests only, no API/integration tests |
| MLflow not in code | docker-compose has MLflow but training ignores it |
| No model explainability | Black-box predictions, no SHAP |
| No structured logging | Console-only via Rich |
| PLAN.md outdated | References Streamlit/Prophet/TensorFlow |

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Package Manager | uv | latest |
| ML: Tree Models | XGBoost, LightGBM | 2.0+, 4.0+ |
| ML: Deep Learning | PyTorch (MPS on host, CPU in Docker) | 2.2+ |
| ML: Preprocessing | scikit-learn | 1.4+ |
| Experiment Tracking | MLflow (primary, self-hosted) + W&B (optional) | 2.10+ |
| Model Explainability | SHAP | 0.44+ |
| API | FastAPI + uvicorn | 0.109+ |
| UI | Gradio | 4.15+ |
| Data Generation | Faker + NumPy | latest |
| Containers | Docker + Docker Compose | latest |
| Linting | ruff | 0.2+ |
| Testing | pytest + pytest-cov | 7.4+ |

---

## Commands

```bash
# Host development (native, MPS acceleration)
make setup          # uv sync --extra dev
make data           # Generate all 4 synthetic datasets
make train          # Train all 4 models (MPS on Apple Silicon)
make evaluate       # Evaluate all models, save CSV results
make ui             # Launch Gradio web interface (localhost:7860)
make serve          # Launch FastAPI inference server (localhost:8000)
make test           # pytest with coverage report
make lint           # ruff check + format validation
make format         # Auto-format code
make clean          # Remove generated artifacts

# Docker development (containerized, CPU)
make docker-build   # Build API + UI Docker images
make docker-up      # docker compose up -d (MLflow + API + UI)
make docker-down    # docker compose down
make docker-test    # Run test suite inside Docker container
make docker-logs    # Tail all container logs
make docker-clean   # Remove images and volumes

# Full pipeline
make all            # data → train → evaluate (host)
```

Individual scripts:
```bash
uv run python scripts/generate_data.py --problem credit_risk|fraud|housing|timeseries|all
uv run python scripts/train.py --model credit_risk|fraud|price|forecaster|all [--no-wandb]
uv run python scripts/evaluate.py --model credit_risk|fraud|price|forecaster|all
```

---

## Project Structure

```
portfolio-ml-system/
├── configs/                    # YAML hyperparameter configs (never hardcoded)
│   ├── credit_risk.yaml
│   ├── fraud_detection.yaml
│   ├── price_prediction.yaml
│   └── demand_forecasting.yaml
├── src/
│   ├── config.py               # YAML config loader with path resolution
│   ├── device.py               # MPS/CUDA/CPU auto-detection
│   ├── logging_config.py       # [NEW] JSON logging configuration
│   ├── data/                   # Synthetic data generators
│   │   ├── generate_credit_risk.py
│   │   ├── generate_fraud.py
│   │   ├── generate_housing.py
│   │   ├── generate_timeseries.py
│   │   └── preprocess.py
│   ├── features/               # Feature engineering pipelines
│   │   ├── credit_risk_features.py
│   │   ├── fraud_features.py
│   │   ├── housing_features.py
│   │   └── timeseries_features.py
│   ├── models/                 # Model architectures (no training logic)
│   │   ├── credit_risk_model.py    # XGBoost classifier factory
│   │   ├── fraud_autoencoder.py    # PyTorch symmetric autoencoder
│   │   ├── lstm_forecaster.py      # PyTorch LSTM with FC head
│   │   └── price_model.py         # LightGBM regressor factory
│   ├── training/               # Training pipeline
│   │   ├── trainer.py              # BaseTrainer (W&B + MLflow + checkpointing)
│   │   ├── train_credit_risk.py
│   │   ├── train_fraud.py
│   │   ├── train_price.py
│   │   └── train_forecaster.py
│   ├── evaluation/             # Metrics computation
│   │   ├── classification_metrics.py
│   │   ├── regression_metrics.py
│   │   ├── timeseries_metrics.py
│   │   └── evaluator.py
│   ├── explainability/         # [NEW] Model explanation module
│   │   ├── __init__.py
│   │   ├── shap_explainer.py       # SHAP for tree models
│   │   └── gradient_explainer.py   # Gradient-based for autoencoder
│   └── serving/                # Inference + API
│       ├── predictor.py            # ModelPredictor (checkpoint loading + inference)
│       └── api.py                  # FastAPI server
├── scripts/                    # CLI entry points
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── serve.py
│   └── run_all.py
├── app/
│   └── gradio_app.py           # Unified Gradio UI (6 tabs after explainability)
├── tests/
│   ├── conftest.py             # Fixtures: tiny models, test data, API client
│   ├── test_data_generation.py # Parametrized generator tests
│   ├── test_features.py        # Parametrized feature engineering tests
│   ├── test_models.py          # Model architecture + shape tests
│   ├── test_serving.py         # ModelPredictor integration tests
│   ├── test_api_endpoints.py   # [NEW] FastAPI TestClient integration
│   ├── test_training_pipeline.py # [NEW] End-to-end training tests
│   ├── test_explainability.py  # [NEW] SHAP + gradient explainer tests
│   └── test_logging.py         # [NEW] Structured logging validation
├── checkpoints/                # Model weights + metadata (gitignored)
├── results/                    # CSV evaluation results (committed)
├── data/                       # Raw + processed data (gitignored)
├── Dockerfile.api              # [NEW] Multi-stage FastAPI container
├── Dockerfile.ui               # [NEW] Multi-stage Gradio container
├── .dockerignore               # [NEW] Exclude data/, .git/, wandb/
├── docker-compose.yml          # [REWRITTEN] Self-contained dev stack
├── docker-compose.prod.yml     # [RENAMED] Production stack (PostgreSQL, Traefik)
├── Makefile                    # [UPDATED] Docker targets added
├── pyproject.toml              # [UPDATED] mlflow, shap, pytest-cov config
├── .env.example                # Environment variable template
├── SPEC.md                     # [NEW] This file
├── decision.md                 # [NEW] Design decisions with rationale
├── CLAUDE.md                   # Project instructions for Claude Code
├── AGENTS.md                   # Agent role definitions
├── PORTS.md                    # Port allocation (3070, 8070, 5070)
└── PLAN.md                     # [DEPRECATED] Replaced by SPEC.md
```

---

## Code Style

Existing codebase follows consistent Python conventions. All new code matches:

```python
"""Module docstring — one-line summary of what this module does."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import load_config, get_project_root

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw data into model-ready features.

    Uses domain-specific transformations per problem type.
    """

    def __init__(self, config: dict):
        self.config = config
        self._scaler = StandardScaler()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering pipeline."""
        df = self._add_derived_features(df)
        df = self._encode_categoricals(df)
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ratio and binning features."""
        df["ratio"] = df["numerator"] / df["denominator"].clip(lower=1)
        return df
```

**Key conventions:**
- Type hints on all function signatures
- Module-level `logger = logging.getLogger(__name__)` (not Rich console for new code)
- snake_case for functions/variables, PascalCase for classes
- 100-char line length (ruff config)
- ruff rules: E, F, I, W
- Import xgboost before torch in any module that uses both (macOS libomp fix)
- All hyperparameters from YAML configs, never hardcoded

---

## Testing Strategy

**Framework:** pytest + pytest-cov

**Test location:** `tests/` directory, mirroring src/ structure

**Coverage target:** 80% of `src/` (measured by `pytest --cov=src --cov-report=term-missing`)

**Test levels:**

| Level | What | Framework | Where |
|-------|------|-----------|-------|
| Unit | Data generators, feature engineering, metrics, logging | pytest, parametrize | `tests/test_data_generation.py`, `test_features.py`, `test_logging.py` |
| Component | Model architectures (shape, gradient, device) | pytest, torch fixtures | `tests/test_models.py` |
| Integration | API endpoints, ModelPredictor, training pipeline | TestClient, tiny fixtures | `tests/test_api_endpoints.py`, `test_serving.py`, `test_training_pipeline.py` |
| Explainability | SHAP values, gradient attribution | pytest, tiny models | `tests/test_explainability.py` |

**Fixture strategy:**
- Tiny real models (50 rows of data, 2 epochs) created in `conftest.py`
- Saved to `tmp_path` (automatic cleanup)
- No mocks for model inference — catches real shape/device bugs
- TestClient for API tests with tiny checkpoints injected
- Import ordering enforced: xgboost before torch (conftest.py)

**Test naming:** `test_<action>_<scenario>_<expected>` (e.g., `test_predict_credit_risk_returns_valid_score`)

**pytest configuration (pyproject.toml):**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "--cov=src --cov-report=term-missing --tb=short -q"
```

---

## Vertical Slices

Each slice delivers end-to-end value and uses TDD (RED → GREEN → REFACTOR).

### Slice 1: Docker Foundation + Dev Workflow

**What it delivers:** `docker compose up` starts MLflow + API + UI from scratch, zero external dependencies.

**Acceptance criteria:**
- [ ] `Dockerfile.api` builds successfully (multi-stage, Python 3.11-slim, CPU PyTorch)
- [ ] `Dockerfile.ui` builds successfully (multi-stage, Python 3.11-slim)
- [ ] `.dockerignore` excludes data/, .git/, wandb/, checkpoints/ from build context
- [ ] `docker-compose.yml` is self-contained (SQLite MLflow, no PostgreSQL, no external networks)
- [ ] Current production compose preserved as `docker-compose.prod.yml`
- [ ] `docker compose up --build` starts 3 services: mlflow (:5070), ml-api (:8070), ml-ui (:3070)
- [ ] Health checks pass on all 3 services within 60 seconds
- [ ] `curl http://localhost:8070/health` returns `{"status": "ok"}`
- [ ] `curl http://localhost:8070/predict/credit-risk` returns a valid prediction (using default Pydantic values)
- [ ] Makefile has docker-build, docker-up, docker-down, docker-logs targets
- [ ] Container images use non-root user

**Files to create:** `Dockerfile.api`, `Dockerfile.ui`, `.dockerignore`
**Files to modify:** `docker-compose.yml` (rewrite), `Makefile` (add docker targets)
**Files to rename:** `docker-compose.yml` → `docker-compose.prod.yml` (current one)

**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

---

### Slice 2: Test Infrastructure + 80% Coverage

**What it delivers:** Comprehensive test suite with coverage report proving code quality.

**Acceptance criteria:**
- [ ] `conftest.py` has fixtures for: tiny trained models (all 4), test DataFrames, tmp checkpoint dirs
- [ ] Parametrized tests for all 4 data generators (shape, types, value ranges, distributions)
- [ ] Parametrized tests for all 4 feature engineering pipelines (output columns, derived features)
- [ ] Model architecture tests: autoencoder (input→output shape, reconstruction error shape), LSTM (batch→forecast shape)
- [ ] API integration tests: all 4 prediction endpoints (happy path, validation errors), /health, /models
- [ ] Training pipeline integration test: generate small data → train → evaluate → checkpoint saved
- [ ] `pytest --cov=src` reports 80%+ coverage
- [ ] `make test` runs full suite with coverage summary
- [ ] No test takes longer than 10 seconds individually
- [ ] All tests pass in Docker (`make docker-test`)

**Files to create:** `tests/test_api_endpoints.py`, `tests/test_training_pipeline.py`
**Files to modify:** `tests/conftest.py`, existing test files (add parametrize), `pyproject.toml` (pytest-cov config)

**Skills:** `test-driven-development`, `incremental-implementation`

---

### Slice 3: MLflow Integration

**What it delivers:** Models logged to MLflow with versioning, visible in MLflow UI at :5070.

**Acceptance criteria:**
- [ ] `BaseTrainer` logs parameters, metrics, and artifacts to MLflow on every training run
- [ ] W&B logging remains optional alongside MLflow (dual tracking)
- [ ] Models registered in MLflow model registry with version numbers
- [ ] `metadata.json` includes `mlflow_run_id` and `mlflow_model_version`
- [ ] `/models` API endpoint includes MLflow version info
- [ ] MLflow UI at http://localhost:5070 shows experiment runs and registered models
- [ ] `mlflow` added to pyproject.toml dependencies
- [ ] Tests verify MLflow logging (using temp tracking URI)
- [ ] Graceful fallback if MLflow server is unreachable

**Files to modify:** `src/training/trainer.py`, `src/serving/predictor.py`, `src/serving/api.py`, `pyproject.toml`
**Files to create:** `tests/test_mlflow_integration.py`

**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

---

### Slice 4: Model Explainability

**What it delivers:** SHAP plots for tree models, gradient-based importance for autoencoder, visible in API + UI.

**Acceptance criteria:**
- [ ] `src/explainability/shap_explainer.py` computes SHAP values for credit risk (XGBoost) and price prediction (LightGBM)
- [ ] `src/explainability/gradient_explainer.py` computes gradient-based feature importance for fraud autoencoder
- [ ] `POST /explain/credit-risk` returns top feature importances + SHAP values per feature
- [ ] `POST /explain/price` returns same for price prediction
- [ ] `POST /explain/fraud` returns gradient-based feature ranking
- [ ] Gradio UI shows feature importance bar charts below predictions (credit risk, price, fraud tabs)
- [ ] `shap` added to pyproject.toml dependencies
- [ ] Tests verify SHAP output shapes, feature names, and API response structure
- [ ] Demand forecasting excluded (LSTM temporal attention out of scope)

**Files to create:** `src/explainability/__init__.py`, `src/explainability/shap_explainer.py`, `src/explainability/gradient_explainer.py`, `tests/test_explainability.py`
**Files to modify:** `src/serving/api.py`, `src/serving/predictor.py`, `app/gradio_app.py`, `pyproject.toml`

**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`

---

### Slice 5: Structured Logging + Production Polish

**What it delivers:** JSON logs in Docker, request/response logging, enriched health checks.

**Acceptance criteria:**
- [ ] `src/logging_config.py` provides `setup_logging()` with JSON formatter for Docker, human-readable for terminal
- [ ] `LOG_FORMAT` env var switches between `json` and `text` (default: `json` in Docker, `text` on host)
- [ ] BaseTrainer uses `logger.info()` instead of `console.print()` for pipeline steps
- [ ] FastAPI middleware logs request method, path, status code, and latency for every request
- [ ] `/health` endpoint returns model availability status (which models have checkpoints)
- [ ] Training run logs include: problem name, dataset size, training time, final metrics
- [ ] Rich console output preserved for Gradio UI only (user-facing)
- [ ] Tests validate JSON log format, middleware behavior, health check response structure
- [ ] Docker logs (`docker compose logs ml-api`) show structured JSON entries

**Files to create:** `src/logging_config.py`, `tests/test_logging.py`
**Files to modify:** `src/training/trainer.py`, `src/serving/api.py`, `scripts/serve.py`

**Skills:** `incremental-implementation`, `test-driven-development`

---

## Boundaries

### Always Do
- Run `make test` before every commit
- Follow existing naming conventions and code patterns
- Keep all hyperparameters in YAML configs
- Use `conftest.py` for shared test fixtures
- Import xgboost before torch in any file using both
- Use port allocations from PORTS.md (3070, 8070, 5070)
- Write the failing test first (RED), then implementation (GREEN)
- Docker containers run as non-root user

### Ask First
- Adding new Python dependencies (check if stdlib alternative exists)
- Changing model architectures or hyperparameters
- Modifying existing API response schemas (backward compatibility)
- Changing port allocations
- Adding environment variables to `.env.example`

### Never Do
- Commit secrets, API keys, or credentials
- Hardcode hyperparameters in Python code (use YAML configs)
- Use MPS-specific code paths in Docker (CPU only)
- Remove or skip failing tests without understanding why
- Break the xgboost-before-torch import ordering
- Use ports outside the 3070-3079 / 8070-8079 range

---

## Success Criteria

When all 5 slices are complete, the system passes these acceptance tests:

1. **Docker:** `docker compose up --build` → 3 services healthy within 90 seconds
2. **API:** `curl localhost:8070/health` → returns model availability status
3. **Predictions:** All 4 `/predict/*` endpoints return valid JSON with correct fields
4. **MLflow:** Browse http://localhost:5070 → registered models with version history visible
5. **Explainability:** `/explain/credit-risk` and `/explain/price` return SHAP values
6. **Gradio:** Browse http://localhost:3070 → all tabs functional, explainability charts visible
7. **Tests:** `make test` → 80%+ coverage, zero failures
8. **Logs:** `docker compose logs ml-api` → JSON-formatted log entries
9. **Lint:** `make lint` → zero warnings

---

## Open Questions

None — all design decisions documented in [decision.md](decision.md). Decisions were made independently based on deep analysis of the existing codebase, portfolio goals, and production ML best practices.

---
---

# Spec: Phase A.1 — Data Streaming Foundation

> **Parent plan:** `~/.claude/plans/lexical-purring-nebula.md` §Phase A.1
> **Depends on:** Completed production-hardening work (Slices 1–5 above)
> **Unlocks:** All remaining slices (A.2–A.9, B.1–B.4, C) that introduce 16 new models across 6 industries

## Objective

Enable training on real Kaggle and Hugging Face datasets **without committing them to the repo or bloating `data/raw/`**. This is the foundation slice that every new industry model depends on.

**Secondary objective (new decision):** Introduce a **three-modality training framework** — every applicable model trains three variants (synthetic-only, stream-only, mixed) so we can pick the best-performing variant per problem for demo. This turns every new model into a small data-ablation study and documents the value of real data vs synthetic.

**Target user:** Armando, building industry-specific demos. Also the Docker build — containers must not ship any real Kaggle data; they must stream or fail gracefully.

**Success looks like:**
- `src/data/stream.py` provides reusable `hf_stream()` and `kaggle_cached()` utilities
- `price_prediction` can train in any of three modalities via a YAML config flag, with no regression to the existing synthetic pipeline
- `data/raw/` size stays under 50 MB after running all three modalities for price_prediction
- The Kaggle cache lives at `~/.cache/kagglehub/` (outside the repo, outside Docker build context)
- Zero real network calls in the default `make test` run; one opt-in integration test fetches a real (tiny) dataset

---

## Three-Modality Training Framework (New Convention)

Every model that has both a synthetic generator and a real public dataset trains **three variants**, saved as distinct checkpoints:

| Modality | Data source | Checkpoint dir | Use case |
|---|---|---|---|
| `synthetic` | Existing synthetic generator (`src/data/generate_housing.py`) | `checkpoints/price_prediction_synthetic/` | Self-contained demo, no network, preserves current behavior |
| `stream` | Real Kaggle/HF dataset only | `checkpoints/price_prediction_stream/` | Best-case "real data" performance baseline |
| `mixed` | Synthetic + streamed real, concatenated with modality label as feature | `checkpoints/price_prediction_mixed/` | Demonstrates augmentation value; often best in practice when real data is small |

The YAML field `data.source: synthetic | stream | mixed` selects the variant. When the user runs `uv run python scripts/train.py --model price --modality all`, the script trains all three and writes three rows to `results/price_prediction_metrics.csv` (one per modality). The modality with the best test metric is logged as "recommended for demo" in `checkpoints/<problem>_<modality>/metadata.json`.

**Backward compatibility:** If `data.source` is missing from a YAML config, trainers default to `synthetic`. Existing 4 models keep working without config changes.

---

## Commands

Commands introduced or modified by this slice:

```bash
# Install new streaming deps (kagglehub, datasets)
uv sync --extra dev

# Set Kaggle credentials once (write to ~/.kaggle/kaggle.json OR export env vars)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
# OR drop kaggle.json into ~/.kaggle/

# Train price_prediction in all three modalities (new)
uv run python scripts/train.py --model price --modality all

# Train a single modality (new)
uv run python scripts/train.py --model price --modality stream

# Default test run — mocked, no network (unchanged)
make test

# Opt-in integration test that hits Kaggle for real (new)
uv run pytest -m network tests/test_streaming.py

# Verify cache location (new)
du -sh ~/.cache/kagglehub 2>/dev/null || echo "no cache yet"
```

---

## Project Structure

New files (✨) and touched files (🔧):

```
07-Portfolio-ML-System/
├── .env.example                      🔧 add KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN
├── pyproject.toml                    🔧 add kagglehub, datasets deps
├── configs/
│   └── price_prediction.yaml         🔧 add `data.source`, `data.kaggle_slug`, `data.hf_dataset`
├── src/
│   └── data/
│       ├── stream.py                 ✨ hf_stream(), kaggle_cached(), iter_batches()
│       ├── kaggle_credentials.py     ✨ load_kaggle_creds() — env vars or ~/.kaggle/kaggle.json
│       └── modality.py               ✨ load_for_modality(problem, modality, synth_fn, stream_slug)
├── src/training/
│   └── train_price.py                🔧 dispatch on config["data"]["source"]
├── scripts/
│   └── train.py                      🔧 add --modality {synthetic,stream,mixed,all} flag
└── tests/
    └── test_streaming.py             ✨ unit tests (mocked) + one @pytest.mark.network test
```

No deletions. Every touched existing file is backward-compatible.

---

## Code Style

### `stream.py` public surface — example

```python
"""Streaming data loaders for Kaggle and Hugging Face datasets.

Keeps external datasets out of the repo by caching to ~/.cache/
(kagglehub) or streaming in-memory (HF datasets).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


def kaggle_cached(slug: str, *, filename: str | None = None) -> Path:
    """Download a Kaggle dataset to ~/.cache/kagglehub/ and return the path.

    Args:
        slug: Kaggle dataset slug, e.g. "arianazmoudeh/airbnbopendata".
        filename: Optional specific file inside the dataset. If None, returns the directory.

    Returns:
        Path to the cached dataset (file or directory).

    Raises:
        RuntimeError: If Kaggle credentials are missing.
    """
    import kagglehub
    path = Path(kagglehub.dataset_download(slug))
    return path / filename if filename else path


def hf_stream(dataset_id: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """Stream rows from a Hugging Face dataset without downloading.

    Args:
        dataset_id: HF dataset identifier, e.g. "lex_glue".
        split: Dataset split name.

    Yields:
        Dict per row.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split, streaming=True)
    yield from ds


def iter_batches(
    source: Iterator[dict[str, Any]] | pd.DataFrame,
    batch_size: int = 1024,
) -> Iterator[pd.DataFrame]:
    """Chunk any row-iterable or DataFrame into DataFrame batches."""
    ...  # implementation
```

### `load_for_modality` — example dispatcher

```python
def load_for_modality(
    modality: str,
    *,
    synthetic_loader: Callable[[], pd.DataFrame],
    stream_slug: str | None = None,
    stream_file: str | None = None,
    stream_adapter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame matching the requested modality.

    Modalities:
        synthetic: call synthetic_loader()
        stream: kaggle_cached(stream_slug) -> read_csv -> stream_adapter()
        mixed: concat(synthetic, stream) with a 'modality' column added
    """
    if modality == "synthetic":
        return synthetic_loader()
    if modality == "stream":
        df = pd.read_csv(kaggle_cached(stream_slug, filename=stream_file))
        return stream_adapter(df) if stream_adapter else df
    if modality == "mixed":
        synth = synthetic_loader().assign(modality="synthetic")
        real = load_for_modality("stream", synthetic_loader=synthetic_loader,
                                 stream_slug=stream_slug, stream_file=stream_file,
                                 stream_adapter=stream_adapter).assign(modality="stream")
        return pd.concat([synth, real], ignore_index=True)
    raise ValueError(f"Unknown modality: {modality}")
```

### Style rules
- Type hints required on every public function
- `from __future__ import annotations` at the top of every new Python file
- Module docstring explaining purpose
- All imports of heavy deps (`kagglehub`, `datasets`) happen **inside** functions — keeps import-time fast and lets mocks patch cleanly
- No print statements — use `logging.getLogger(__name__)`

---

## Testing Strategy

- Framework: `pytest` (existing)
- Tests in `tests/test_streaming.py`

**Required tests (all mocked, run in `make test`):**

1. `test_kaggle_cached_returns_path` — monkeypatch `kagglehub.dataset_download` → returns tmp dir → assert the path
2. `test_kaggle_cached_with_filename` — same as above + filename param → returns `dir/filename`
3. `test_hf_stream_yields_rows` — monkeypatch `datasets.load_dataset` → returns fake iterable → assert iteration works
4. `test_iter_batches_chunks_correctly` — input 2500 rows, batch_size=1000 → yields 3 DataFrames (1000, 1000, 500)
5. `test_load_for_modality_synthetic` — calls synthetic_loader, returns its output
6. `test_load_for_modality_stream` — mocked kaggle_cached, returns real shape
7. `test_load_for_modality_mixed` — returns concat with `modality` column
8. `test_load_for_modality_invalid_raises` — raises ValueError for unknown modality
9. `test_load_kaggle_creds_from_env` — monkeypatch env vars → loader reads them
10. `test_load_kaggle_creds_from_file` — monkeypatch HOME to tmp dir with kaggle.json → loader reads it
11. `test_price_trainer_respects_modality_flag` — integration: run PricePredictionTrainer with each modality using tiny mocked data → all three checkpoints written
12. `test_missing_data_source_defaults_to_synthetic` — backward compatibility guard

**Integration test (opt-in via `-m network`):**

13. `@pytest.mark.network test_kaggle_cached_real_tiny_dataset` — pulls a tiny Kaggle dataset (e.g., iris-style <1 MB) → verifies the actual Kaggle API works. Skipped in CI unless `KAGGLE_USERNAME`+`KAGGLE_KEY` set.

**Coverage target:** 90%+ for `src/data/stream.py`, `src/data/modality.py`, `src/data/kaggle_credentials.py`.

**pytest config addition** (pyproject.toml):
```toml
[tool.pytest.ini_options]
markers = [
    "network: tests that hit real external APIs (skipped by default)",
]
```

---

## Boundaries

### Always
- Import `kagglehub` / `datasets` **inside** functions, never at module top-level — keeps mocking clean and import-time fast
- Cache to `~/.cache/kagglehub/` (outside the repo) — never write datasets into `data/raw/`
- Use typed function signatures on all public surface
- Respect existing BaseTrainer contract — streaming is opt-in and backward-compatible
- Mock all external libs in unit tests; CI never hits real Kaggle

### Ask first
- Adding new dependencies beyond `kagglehub` and `datasets` (e.g., `huggingface_hub`, `pyarrow`)
- Changing existing trainer interfaces — only add kwargs, never remove or rename
- Committing real Kaggle data to the repo (answer is always no; confirm intent if tempted)
- Changing modality semantics or checkpoint directory naming convention after first commit

### Never
- Commit `kaggle.json` or any file containing `KAGGLE_KEY`
- Hardcode Kaggle credentials in code or tests
- Bypass the cache — re-downloading a 3 M-row dataset on every train run is unacceptable
- Break the three-modality contract once a model adopts it — `checkpoints/<problem>_<modality>/` must be stable
- Import `torch` before `xgboost` / `lightgbm` in new files (preserves the existing libomp fix)

---

## Success Criteria (Phase A.1 acceptance)

1. `uv sync` installs `kagglehub` and `datasets` without errors on Apple Silicon
2. `tests/test_streaming.py` passes with ≥90 % coverage of the three new modules — `make test` stays green, total coverage ≥80 %
3. `uv run python scripts/train.py --model price --modality synthetic` produces `checkpoints/price_prediction_synthetic/model.pkl` and `metadata.json` — byte-for-byte reproducible given the same seed
4. `uv run python scripts/train.py --model price --modality stream` downloads Zillow data to `~/.cache/kagglehub/`, produces `checkpoints/price_prediction_stream/`, and `data/raw/` grows by < 1 MB
5. `uv run python scripts/train.py --model price --modality mixed` produces `checkpoints/price_prediction_mixed/` with the `modality` feature column correctly present
6. `uv run python scripts/train.py --model price --modality all` runs all three sequentially, writes three rows to `results/price_prediction_metrics.csv`, and logs a "recommended" modality based on test R²
7. `make lint` → zero warnings
8. `pytest -m network tests/test_streaming.py` passes when Kaggle creds are set (manually verified once; not run in CI)
9. All four existing models (credit_risk, fraud_detection, price_prediction synthetic path, demand_forecasting) still train and pass their existing tests — no regressions
10. `du -sh data/raw/` stays under 50 MB after full run

---

## Open Questions

- **Q1 — Zillow dataset schema vs synthetic housing schema:** Need to write a `stream_adapter` that renames Zillow columns to match our existing `housing.csv` schema (`square_feet`, `bedrooms`, etc.). This is expected A.1 scope; concrete column map will be decided at `/build` time after inspecting the real Kaggle CSV.
- **Q2 — Checkpoint dir breaking change:** Existing checkpoint lives at `checkpoints/price_prediction/` (no modality suffix). Migration options: (a) keep that path as alias for `_synthetic`, (b) rename during A.1 and update predictor. Deferred to `/plan` step — likely go with (a) for zero-downtime migration.
- **Q3 — W&B / MLflow run naming:** Do the three modalities become three separate MLflow runs under one experiment, or three child runs under a parent? Deferred to `/plan` step.

---
---

# Spec: Phase A.2 — Next.js Frontend Scaffolding

> **Parent plan:** `~/.claude/plans/lexical-purring-nebula.md` §Phase A.2
> **Depends on:** Phase A.1 complete (provides `/predict/*` + `/explain/*` backend endpoints)
> **Unlocks:** Phases A.3–A.8 (6 industry slices each add pages to this frontend in parallel)

## Objective

Replace the Gradio UI with a production-grade **Next.js 14 + shadcn/ui** web app that looks like a product, not a research prototype. This slice is the scaffold — a landing page with 6 industry tiles, one proof-of-concept model page (credit risk) exercising the full form → prediction → explainability loop, and a new Dockerized `ml-web:3071` service. Subsequent slices fan out industry-specific pages on top of this foundation.

**Target users:**
1. **Prospective clients** Armando demos to (dental clinic owners, healthcare CIOs, fintech PMs, etc.) — they should see a polished product UI
2. **Recruiters / hiring managers** browsing the public deploy — it should look like something built in production, not a toy
3. **Armando himself** during development — fast HMR via local `next dev`, clean component abstractions

**Success looks like:**
- `pnpm dev` inside `web/` → HMR at `http://localhost:3071` with landing tiles + working credit-risk model page
- `make docker-up` → prod-built `ml-web` container at `http://localhost:3071` alongside existing `ml-api:8070` + `mlflow:5070`
- Credit risk page accepts form inputs (income, credit score, etc.), submits to FastAPI, shows risk score + recommendation + SHAP bar chart
- Dark mode toggle, responsive layout, accessibility-passing shadcn defaults
- Component tests with Vitest cover the 3 reusable components; lint + typecheck clean
- No backend changes — existing `/predict/credit-risk` + `/explain/credit-risk` endpoints are consumed as-is

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Framework | **Next.js 14 (App Router)** | Server Components + streaming, modern React patterns |
| Language | **TypeScript strict mode** | Compile-time safety on API client + forms |
| Styling | **TailwindCSS v3** | Utility-first, shadcn/ui default baseline |
| Components | **shadcn/ui** (CLI-installed) | Owned source code under `components/ui/`, not a versioned dep |
| Data fetching | **TanStack Query v5** (`@tanstack/react-query`) | Caching, loading/error states, retries out of the box |
| Form validation | **Zod** + `@hookform/resolvers/zod` + **react-hook-form** | Typed forms with minimal boilerplate |
| Charts | **Recharts** | SHAP bar chart for `/explain/*` responses |
| Icons | **lucide-react** | shadcn default |
| Package manager | **pnpm** | Fast, content-addressable lockfile |
| Tests | **Vitest** + **@testing-library/react** + **jsdom** | Fast component tests; Playwright deferred to Phase C |
| Lint/format | **ESLint** (next config) + **Prettier** (with Tailwind plugin) | Next.js defaults |
| Node runtime | **Node 20 LTS** | matches Next 14 requirements |

No tRPC, no server actions for form submit — keep the initial scaffold simple. TanStack Query + typed fetch client is sufficient.

---

## Commands

Local dev (outside Docker — fastest iteration, preferred during development):

```bash
# One-time setup
cd web
pnpm install

# Start dev server at http://localhost:3071 (HMR enabled)
NEXT_PUBLIC_API_URL=http://localhost:8070 pnpm dev

# Or via root Makefile target:
make web-dev

# Type check
pnpm typecheck          # runs `tsc --noEmit`

# Lint
pnpm lint               # next lint + prettier --check
pnpm lint:fix           # auto-fix

# Vitest component tests
pnpm test               # one-shot
pnpm test:watch         # watch mode
pnpm test:coverage      # with c8 coverage, target ≥80% on components/
```

Docker:

```bash
# Prod build + serve at http://localhost:3071
make docker-up

# Dev mode with HMR + volume mount (slower iteration, works offline)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up web

# Just the web service (prod)
docker compose up ml-web --build
```

Root Makefile additions (new targets):

```make
web-install:    cd web && pnpm install
web-dev:        cd web && NEXT_PUBLIC_API_URL=http://localhost:8070 pnpm dev
web-build:      cd web && pnpm build
web-test:       cd web && pnpm test
web-lint:       cd web && pnpm lint
```

---

## Project Structure

New `web/` directory:

```
web/
├── .env.example                       # NEXT_PUBLIC_API_URL placeholder
├── .eslintrc.json                     # extends next/core-web-vitals + prettier
├── .prettierrc.json                   # Tailwind-aware prettier config
├── .gitignore                         # node_modules, .next, coverage
├── README.md                          # dev + docker quick start
├── package.json
├── pnpm-lock.yaml
├── tsconfig.json                      # strict, paths "@/*": ["./"]
├── next.config.mjs                    # output: 'standalone' for Docker
├── tailwind.config.ts                 # shadcn content globs + theme
├── postcss.config.mjs
├── components.json                    # shadcn config (generated by CLI)
├── vitest.config.ts                   # jsdom env + coverage
├── vitest.setup.ts                    # @testing-library/jest-dom extensions
├── app/
│   ├── layout.tsx                     # Root layout: nav, theme provider, QueryClient provider
│   ├── page.tsx                       # Landing page: 6 industry tiles
│   ├── globals.css                    # Tailwind directives + shadcn CSS vars
│   ├── providers.tsx                  # Client: QueryClientProvider, ThemeProvider wrapper
│   └── fintech/
│       └── credit-risk/
│           └── page.tsx               # PoC model page (client component)
├── components/
│   ├── ui/                            # shadcn components: button, card, input, label, slider, switch, form, toaster, ...
│   ├── ModelForm.tsx                  # Reusable form wrapper (takes Zod schema + fields config)
│   ├── PredictionResult.tsx           # Score/recommendation/confidence card
│   ├── ExplainabilityChart.tsx        # Recharts bar chart for SHAP values
│   ├── IndustryTile.tsx               # Landing-page industry card
│   ├── ThemeToggle.tsx                # Light/dark switch with localStorage persistence
│   └── Nav.tsx                        # Top navigation
├── lib/
│   ├── api.ts                         # Typed FastAPI client: predictCreditRisk, explainCreditRisk, ...
│   ├── schemas.ts                     # Zod schemas per model input
│   ├── query-client.ts                # TanStack QueryClient singleton + default options
│   └── utils.ts                       # shadcn `cn()` helper (generated)
└── __tests__/
    └── components/
        ├── ModelForm.test.tsx
        ├── PredictionResult.test.tsx
        └── ExplainabilityChart.test.tsx
```

New top-level files:
- `Dockerfile.web` — multi-stage node:20-alpine build using `pnpm` + Next.js standalone output
- `docker-compose.yml` — add `ml-web` service with port 3071, depends_on `ml-api`
- `docker-compose.dev.yml` — override layer: mount `web/` as volume, run `pnpm dev` instead of `pnpm start`

---

## Code Style

### Typed API client — `web/lib/api.ts`

```typescript
import { z } from "zod";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8070";

// Response schemas — decode + validate FastAPI output at the boundary
export const CreditRiskPredictionSchema = z.object({
  probability_default: z.number().min(0).max(1),
  risk_score: z.number(),
  recommendation: z.enum(["APPROVE", "REVIEW", "DECLINE"]),
  confidence: z.number().min(0).max(1),
});
export type CreditRiskPrediction = z.infer<typeof CreditRiskPredictionSchema>;

export const ExplanationSchema = z.object({
  feature_importances: z.record(z.string(), z.number()),
  base_value: z.number(),
});
export type Explanation = z.infer<typeof ExplanationSchema>;

export class ApiError extends Error {
  constructor(public status: number, public body: unknown) {
    super(`API ${status}`);
  }
}

async function post<T>(path: string, body: unknown, schema: z.ZodSchema<T>): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await res.json().catch(() => null));
  return schema.parse(await res.json());
}

export const predictCreditRisk = (input: CreditRiskInput) =>
  post("/predict/credit-risk", input, CreditRiskPredictionSchema);

export const explainCreditRisk = (input: CreditRiskInput) =>
  post("/explain/credit-risk", input, ExplanationSchema);
```

### Credit-risk page — composition, not duplication

```tsx
"use client";

import { useMutation } from "@tanstack/react-query";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { ModelForm } from "@/components/ModelForm";
import { PredictionResult } from "@/components/PredictionResult";
import { ExplainabilityChart } from "@/components/ExplainabilityChart";
import { CreditRiskInputSchema, type CreditRiskInput } from "@/lib/schemas";
import { predictCreditRisk, explainCreditRisk } from "@/lib/api";

export default function CreditRiskPage() {
  const form = useForm<CreditRiskInput>({ resolver: zodResolver(CreditRiskInputSchema) });
  const predict = useMutation({ mutationFn: predictCreditRisk });
  const explain = useMutation({ mutationFn: explainCreditRisk });

  const onSubmit = async (data: CreditRiskInput) => {
    await Promise.all([predict.mutateAsync(data), explain.mutateAsync(data)]);
  };

  return (
    <main className="container grid gap-6 md:grid-cols-2 py-8">
      <ModelForm form={form} onSubmit={onSubmit} fields={CREDIT_RISK_FIELDS} />
      <div className="space-y-4">
        {predict.data && <PredictionResult result={predict.data} />}
        {explain.data && <ExplainabilityChart importances={explain.data.feature_importances} />}
      </div>
    </main>
  );
}
```

### Style rules
- Client components explicitly marked `"use client"` at the top; everything else is a Server Component by default
- No `any` without a `// eslint-disable-next-line` + justification comment
- Zod schemas live in `lib/schemas.ts`; API response schemas live in `lib/api.ts` alongside the call
- Paths via `@/` alias (tsconfig `paths`)
- Tailwind classes via `cn()` utility (shadcn default) for conditional merging
- Error handling: API errors surface as `ApiError` (with status + body); React Query handles retry/backoff
- Accessibility: shadcn components are a11y-ready by default; don't override ARIA unless there's a reason

---

## Testing Strategy

### Vitest — component tests

Location: `web/__tests__/components/*.test.tsx`

Target: ≥80% line coverage on `web/components/` (the reusable logic — not the page-level compositions).

Required tests in A.2:
- `ModelForm.test.tsx` — renders given field config, validates via Zod schema, calls `onSubmit` with parsed values, surfaces validation errors inline
- `PredictionResult.test.tsx` — renders score/recommendation/confidence, applies appropriate color for APPROVE/REVIEW/DECLINE, rounds probability to 2 dp
- `ExplainabilityChart.test.tsx` — renders a Recharts BarChart with one bar per feature, sorted by absolute importance descending, bars colored by sign

No network in tests — `lib/api.ts` is mocked via `vi.mock("@/lib/api")` at the test level.

### Typecheck gate

`pnpm typecheck` must pass in CI. Strict TS, no implicit `any`.

### Manual smoke check (required before calling A.2 done)

```bash
# 1) Local dev smoke
make web-dev                  # HMR server at :3071
# Visit http://localhost:3071, see 6 industry tiles
# Click Fintech → Credit Risk
# Fill the form, submit, see predicted risk score + SHAP bar chart
# Toggle dark mode, verify theme persists on reload

# 2) Docker smoke
make docker-up                # 4 services: mlflow, ml-api, ml-web, (ml-ui legacy still running)
# Browse http://localhost:3071 — same experience
# Verify ml-web can reach ml-api inside compose network
```

Playwright E2E is out of scope for A.2 — planned for Phase C.

---

## Boundaries

### Always
- Consume backend via `@/lib/api.ts` typed functions — never inline `fetch(url)` in a page
- Validate all external JSON with Zod at the API boundary — no `as` casts on fetch responses
- Use shadcn/ui components for form, card, button primitives; compose Tailwind utilities on top
- Run `pnpm typecheck` + `pnpm lint` + `pnpm test` before committing frontend changes
- Use env var `NEXT_PUBLIC_API_URL` for the FastAPI base URL — never hardcode `localhost:8070`
- Write component tests for any reusable component in `web/components/` (excluding shadcn primitives)

### Ask first
- Adding a dep beyond the list in Tech Stack above (e.g., framer-motion, date-fns, new icon library)
- Changing the Next.js router mode (App → Pages) — App Router is locked in
- Adding authentication — not in A.2 scope
- Server Actions for form submission — A.2 uses client mutations
- Swapping package manager away from pnpm
- Adding analytics / telemetry

### Never
- Bypass the typed API client with inline fetch in a component
- Use `any` without a disable comment justifying it
- Commit `web/.next/`, `web/node_modules/`, or `web/coverage/` (gitignored)
- Hardcode backend URLs
- Import server-only code into client components (and vice versa)
- Delete or modify `app/gradio_app.py` — Gradio retires in A.9, not A.2 (parallel operation enforced)
- Add new FastAPI routes — A.2 is frontend-only; backend is read-only

---

## Success Criteria (Phase A.2 acceptance)

1. `cd web && pnpm install` succeeds from a clean clone
2. `pnpm dev` serves http://localhost:3071 with HMR — landing page renders with 6 industry tiles
3. Clicking the Fintech tile navigates to `/fintech/credit-risk` with a working form
4. Submitting the credit risk form triggers `POST /predict/credit-risk` + `POST /explain/credit-risk` (verified via browser DevTools Network)
5. Response renders: risk score, APPROVE/REVIEW/DECLINE recommendation with color, confidence, Recharts bar chart of top features
6. Dark mode toggle works and persists across reload via `localStorage`
7. Responsive: renders correctly at 375px, 768px, 1280px viewports
8. `pnpm typecheck` passes (strict mode, zero errors)
9. `pnpm lint` passes (zero warnings)
10. `pnpm test` passes with ≥80% coverage on `web/components/`
11. `make docker-up` brings up `ml-web:3071` alongside existing services; container serves the same UI
12. `docker compose -f docker-compose.yml -f docker-compose.dev.yml up web` serves HMR-enabled dev container
13. Existing Python test suite still green (no backend changes): `make test` → 323+ pass, 90%+ coverage
14. No backend code modified — `git diff main src/` shows zero changes
15. Gradio (`ml-ui`) still runs in parallel — A.2 does not remove it (A.9 does)

---

## Parallelization Strategy (for A.3–A.8 fan-out)

Once A.2 lands, the 6 industry slices (A.3–A.8) can run in parallel worktrees per the strategy locked in `tasks/plan.md`. Each industry slice adds:
- `web/app/<industry>/<model>/page.tsx` — copy-paste from the credit-risk PoC, swap schema + fields + API call
- `web/lib/schemas.ts` — add the new input schema
- `web/lib/api.ts` — add the new typed API function
- `src/data/`, `src/features/`, `src/training/`, etc. — backend model additions

These paths are disjoint across the 6 industries except for `web/lib/api.ts` and `web/lib/schemas.ts`, which each slice appends to. A.2 must leave those files structured so appends are clean (one function per model, grouped comments by industry).

---

## Open Questions

- **Q-A.2-1 — Landing page copy:** who writes the taglines for each industry tile? For A.2, I'll use placeholder text ("Real-estate price & rental estimators") that Armando can tune later. Marked `TODO(copy)` in the source.
- **Q-A.2-2 — Vercel deploy:** deferred to Phase C. A.2 only ensures Next.js standalone output works locally + in Docker; Vercel-specific config (`vercel.json`, env var mapping) is Phase C.
- **Q-A.2-3 — Analytics / error tracking:** no Plausible / PostHog / Sentry in A.2. Revisit at Phase C if public deploy goes live.