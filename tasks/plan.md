# Implementation Plan: Portfolio ML System — Production Hardening

## Overview

Break SPEC.md into 22 tasks across 5 vertical slices. Each task uses TDD (RED → GREEN → REFACTOR) and is sized to touch ≤5 files. Tasks are ordered by dependency: Docker foundation first, then tests, then features that build on both.

## Dependency Graph

```
Slice 1: Docker Foundation (Tasks 1.1–1.4)
    │
    ├──→ Slice 2: Test Infrastructure (Tasks 2.1–2.6)
    │        │
    │        ├──→ Slice 3: MLflow Integration (Tasks 3.1–3.4)
    │        │
    │        ├──→ Slice 4: Model Explainability (Tasks 4.1–4.4)
    │        │
    │        └──→ Slice 5: Structured Logging (Tasks 5.1–5.4)
    │
    └──→ Makefile docker targets (used by Slice 2 docker-test)
```

Cross-slice dependencies:
- Slice 2 needs Slice 1 for `make docker-test`
- Slice 3 needs Slice 1 for MLflow container + Slice 2 for test fixtures
- Slice 4 needs Slice 2 for test fixtures (tiny models)
- Slice 5 needs Slice 2 for test patterns

## Architecture Decisions

- See [decision.md](../decision.md) for all 7 decisions with rationale
- Key: MLflow primary + W&B optional, CPU-only Docker, tiny real model fixtures, SHAP + gradients

---

## Slice 1: Docker Foundation + Dev Workflow

**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`
**Delivers:** `docker compose up` starts MLflow + API + UI from scratch

### Task 1.1: Create .dockerignore and Rename Production Compose

**Description:** Prepare the Docker build context by excluding unnecessary files, and preserve the existing production compose before we rewrite the default.

**Scope:** XS (2 files)

**Steps:**
- **RED:** N/A (config files, no test target yet)
- **GREEN:** Create `.dockerignore` excluding `data/`, `.git/`, `wandb/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `.env`, `checkpoints/` (checkpoints baked into image separately). Rename current `docker-compose.yml` → `docker-compose.prod.yml`.
- **REFACTOR:** Verify .dockerignore doesn't exclude files needed by Dockerfiles (src/, configs/, app/, pyproject.toml, uv.lock).

**Acceptance criteria:**
- [ ] `.dockerignore` exists with correct exclusions
- [ ] `docker-compose.prod.yml` contains original production config
- [ ] No `docker-compose.yml` exists yet (created in Task 1.3)

**Files:** `.dockerignore` (new), `docker-compose.yml` → `docker-compose.prod.yml` (rename)
**Dependencies:** None

---

### Task 1.2: Create Dockerfile.api (Multi-Stage FastAPI)

**Description:** Build a multi-stage Dockerfile for the FastAPI inference server. Builder stage installs dependencies with uv; runtime stage copies only what's needed.

**Scope:** S (1 file)

**Steps:**
- **RED:** Write a test script `tests/test_docker_build.sh` that runs `docker build -f Dockerfile.api -t ml-api-test .` and asserts exit code 0. Run it — it fails because Dockerfile.api doesn't exist.
- **GREEN:** Create `Dockerfile.api`:
  - Stage 1 (builder): `python:3.11-slim`, install uv, copy `pyproject.toml` + `uv.lock`, run `uv sync` with CPU-only PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`).
  - Stage 2 (runtime): `python:3.11-slim`, copy venv from builder, copy `src/`, `configs/`, `checkpoints/`, `scripts/serve.py`. Create non-root user `mluser`. Expose port 8000. CMD: `python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000`.
- **REFACTOR:** Minimize layer count, add `HEALTHCHECK` instruction, order COPY statements for cache efficiency (deps before source).

**Acceptance criteria:**
- [ ] `docker build -f Dockerfile.api -t ml-api-test .` succeeds
- [ ] Image uses non-root user
- [ ] Image size < 3GB (CPU PyTorch is ~800MB)
- [ ] `HEALTHCHECK` defined

**Files:** `Dockerfile.api` (new)
**Dependencies:** Task 1.1 (.dockerignore must exist)

---

### Task 1.3: Create Dockerfile.ui + Self-Contained docker-compose.yml

**Description:** Build the Gradio UI Dockerfile and create a new self-contained docker-compose.yml that starts MLflow (SQLite), API, and UI with zero external dependencies.

**Scope:** M (3 files)

**Steps:**
- **RED:** Attempt `docker compose up --build -d` — fails because docker-compose.yml doesn't exist.
- **GREEN:**
  1. Create `Dockerfile.ui`: same multi-stage pattern as Dockerfile.api but copies `app/` instead of `scripts/serve.py`, CMD runs `python app/gradio_app.py`. Non-root user, HEALTHCHECK on port 7860 (Gradio default).
  2. Create new `docker-compose.yml`:
     - `mlflow` service: `ghcr.io/mlflow/mlflow:latest`, SQLite backend (`sqlite:////mlflow/mlflow.db`), port 5070:5000, healthcheck.
     - `ml-api` service: build from `Dockerfile.api`, port 8070:8000, depends_on mlflow (healthy), volume mount `checkpoints/` for pre-trained models, healthcheck on `/health`.
     - `ml-ui` service: build from `Dockerfile.ui`, port 3070:7860, depends_on ml-api (healthy), env `ML_API_URL=http://ml-api:8000`, healthcheck.
     - Single bridge network `ml-network` (no external deps).
     - Named volume `mlflow_data` for MLflow persistence.
- **REFACTOR:** Add resource limits (mem_limit), restart policies, log rotation config.

**Acceptance criteria:**
- [ ] `docker compose up --build -d` starts 3 services
- [ ] `docker compose ps` shows all 3 healthy within 90 seconds
- [ ] `curl http://localhost:8070/health` returns `{"status": "ok"}`
- [ ] `curl http://localhost:5070` returns MLflow UI HTML
- [ ] No external network or PostgreSQL dependency

**Files:** `Dockerfile.ui` (new), `docker-compose.yml` (new)
**Dependencies:** Task 1.1, Task 1.2

---

### Task 1.4: Makefile Docker Targets

**Description:** Add Docker-related Make targets for build, up, down, logs, and test.

**Scope:** XS (1 file)

**Steps:**
- **RED:** Run `make docker-build` — fails because target doesn't exist.
- **GREEN:** Add to Makefile:
  ```makefile
  docker-build:
  	docker compose build

  docker-up:
  	docker compose up -d

  docker-down:
  	docker compose down

  docker-logs:
  	docker compose logs -f

  docker-test:
  	docker compose run --rm ml-api python -m pytest tests/ -v --tb=short

  docker-clean:
  	docker compose down -v --rmi local
  ```
- **REFACTOR:** Add `.PHONY` declarations, ensure targets work from project root.

**Acceptance criteria:**
- [ ] `make docker-build` builds both images
- [ ] `make docker-up` starts all services
- [ ] `make docker-down` stops services
- [ ] `make docker-logs` tails logs
- [ ] `make docker-test` runs pytest inside container

**Files:** `Makefile` (modify)
**Dependencies:** Task 1.3

---

### Checkpoint: Slice 1

- [ ] `docker compose up --build` → 3 healthy services
- [ ] `curl http://localhost:8070/health` → `{"status": "ok"}`
- [ ] `curl -X POST http://localhost:8070/predict/credit-risk` → valid prediction
- [ ] `make docker-build`, `make docker-up`, `make docker-down` all work
- [ ] Images use non-root user
- [ ] **Commit and verify `make lint` passes**

---

## Slice 2: Test Infrastructure + 80% Coverage

**Skills:** `test-driven-development`, `incremental-implementation`
**Delivers:** Comprehensive test suite with 80%+ coverage

### Task 2.1: pytest-cov Configuration + Enhanced conftest.py

**Description:** Configure pytest coverage reporting and create comprehensive test fixtures with tiny real models.

**Scope:** M (2 files)

**Steps:**
- **RED:** Run `pytest --cov=src` — fails or shows minimal coverage with no fixtures.
- **GREEN:**
  1. Update `pyproject.toml` `[tool.pytest.ini_options]`:
     ```toml
     addopts = "--cov=src --cov-report=term-missing --tb=short -q"
     ```
  2. Rewrite `tests/conftest.py`:
     - Keep xgboost/lightgbm import ordering
     - Add `@pytest.fixture(scope="session")` for tiny models:
       - `credit_risk_model`: Train XGBClassifier on 100 rows, 5 estimators
       - `fraud_autoencoder`: Train FraudAutoencoder on 100 rows, 2 epochs
       - `price_model`: Train LGBMRegressor on 100 rows, 5 estimators
       - `lstm_model`: Train LSTMForecaster on 50 rows, 2 epochs
     - `@pytest.fixture` for test DataFrames (credit, fraud, housing, timeseries)
     - `@pytest.fixture` for `tmp_checkpoint_dir` that saves tiny models to tmp_path
     - `@pytest.fixture` for `predictor` using tmp checkpoint dir
- **REFACTOR:** Ensure fixtures use `scope="session"` for models (expensive to create, immutable).

**Acceptance criteria:**
- [ ] `pytest --cov=src` runs with coverage report
- [ ] `conftest.py` provides 4 tiny model fixtures + 4 test DataFrame fixtures
- [ ] Fixtures create models in < 5 seconds total
- [ ] Existing tests still pass

**Files:** `pyproject.toml` (modify), `tests/conftest.py` (rewrite)
**Dependencies:** None (can run before Slice 1)

---

### Task 2.2: Parametrized Data Generator Tests

**Description:** Expand data generator tests with parametrize decorators for multiple sample sizes and comprehensive column/distribution validation.

**Scope:** S (1 file)

**Steps:**
- **RED:** Add parametrized tests that check column completeness, dtype correctness, and distribution bounds for each generator. Tests reference columns not yet validated → some fail (or we add new assertions that catch edge cases).
- **GREEN:** Rewrite `tests/test_data_generation.py`:
  - `@pytest.mark.parametrize("n_samples", [100, 1000, 5000])` for credit risk, fraud, housing
  - `@pytest.mark.parametrize("n_years", [1, 2])` for timeseries
  - Assert all expected columns present (match YAML config)
  - Assert dtype correctness (numerical cols are float/int, categoricals are string)
  - Assert value range constraints (age 18-75, credit_score 300-850, etc.)
  - Assert target distribution within expected bounds
- **REFACTOR:** Extract common assertions into helper functions if patterns repeat.

**Acceptance criteria:**
- [ ] Each generator tested with multiple sample sizes
- [ ] Column completeness validated against config
- [ ] Distribution bounds validated (default rate, fraud rate, price positivity)
- [ ] All tests pass with `pytest tests/test_data_generation.py -v`

**Files:** `tests/test_data_generation.py` (rewrite)
**Dependencies:** Task 2.1 (uses test DataFrame fixtures)

---

### Task 2.3: Parametrized Feature Engineering Tests

**Description:** Expand feature engineering tests to validate all derived features, column presence, and value correctness.

**Scope:** S (1 file)

**Steps:**
- **RED:** Add tests checking specific derived feature values (e.g., `loan_to_income = loan_amount / annual_income`). Currently not tested.
- **GREEN:** Rewrite `tests/test_features.py`:
  - Test each feature pipeline: credit_risk, fraud, housing, timeseries
  - Assert all `get_feature_columns()` columns present after `engineer_features()`
  - Assert derived feature correctness with known inputs (e.g., row with income=100k and loan=25k → loan_to_income=0.25)
  - Assert no NaN values in output features
  - Test timeseries `create_sequences()` output shapes: X shape (n, window, 1), y shape (n, horizon)
- **REFACTOR:** Use `@pytest.mark.parametrize` for different input scenarios.

**Acceptance criteria:**
- [ ] All 4 feature pipelines tested
- [ ] Derived feature values verified with known inputs
- [ ] No NaN in output features
- [ ] Timeseries sequence shapes validated
- [ ] All tests pass

**Files:** `tests/test_features.py` (rewrite)
**Dependencies:** Task 2.1 (uses test DataFrame fixtures)

---

### Task 2.4: API Integration Tests with TestClient

**Description:** Write comprehensive API endpoint tests using FastAPI's TestClient. Test happy paths, default values, and error handling.

**Scope:** M (2 files)

**Steps:**
- **RED:** Create `tests/test_api_endpoints.py` with tests for all endpoints:
  - `POST /predict/credit-risk` — returns risk_score, recommendation, confidence
  - `POST /predict/fraud` — returns risk_level, fraud_probability
  - `POST /predict/price` — returns predicted_price, price_range
  - `POST /predict/demand` — returns predictions (7 values), product
  - `GET /health` — returns status
  - `GET /models` — returns model info dict
  Tests fail because they need a predictor with loaded models.
- **GREEN:** Tests use `conftest.py` predictor fixture with tiny checkpoints. Override the `predictor` dependency in api.py using FastAPI's dependency override mechanism or monkeypatch the module-level `predictor` variable. Each test validates response status code + response body structure.
- **REFACTOR:** Group tests by endpoint, use parametrize for different valid inputs.

**Acceptance criteria:**
- [ ] All 4 prediction endpoints tested (happy path)
- [ ] `/health` and `/models` endpoints tested
- [ ] Response status codes validated (200 for success)
- [ ] Response body structure validated (correct keys, correct types)
- [ ] Default Pydantic values produce valid predictions
- [ ] Tests don't require Docker (use TestClient in-process)

**Files:** `tests/test_api_endpoints.py` (new), `tests/conftest.py` (add api_client fixture)
**Dependencies:** Task 2.1 (conftest fixtures)

---

### Task 2.5: Training Pipeline Integration Test

**Description:** Test the full training pipeline end-to-end: generate small data → train → evaluate → checkpoint saved → results CSV written.

**Scope:** M (1 file, but exercises many modules)

**Steps:**
- **RED:** Create `tests/test_training_pipeline.py` with tests that:
  - Instantiate CreditRiskTrainer with tiny data
  - Run the full `trainer.run()` pipeline
  - Assert checkpoint directory created with model.json + metadata.json
  - Assert results CSV written with expected metric keys
  - Test fails initially because training on tiny data may hit edge cases.
- **GREEN:** Write integration tests for at least 2 trainers (CreditRiskTrainer, FraudDetectionTrainer as the most complex). Use small data (100 rows), `--no-wandb`, and temp directories for output. Monkeypatch `get_project_root()` to use tmp_path.
- **REFACTOR:** Extract common pipeline test pattern into a helper.

**Acceptance criteria:**
- [ ] CreditRiskTrainer.run() produces checkpoint + CSV on tiny data
- [ ] FraudDetectionTrainer.run() produces checkpoint (autoencoder.pt, scaler.pkl, isolation_forest.pkl) + CSV
- [ ] Metadata.json contains expected keys (problem, model_type, metrics, timestamp)
- [ ] Tests complete in < 30 seconds total
- [ ] No W&B calls made during tests

**Files:** `tests/test_training_pipeline.py` (new)
**Dependencies:** Task 2.1 (conftest fixtures, tmp_path patterns)

---

### Task 2.6: Coverage Audit + Gap Filling

**Description:** Run coverage report, identify uncovered lines, and add targeted tests to reach 80%.

**Scope:** M (2-3 files)

**Steps:**
- **RED:** Run `pytest --cov=src --cov-report=term-missing`. Identify modules below 80%.
- **GREEN:** Write targeted tests for uncovered code paths:
  - `src/config.py` — test `load_config()` with valid/invalid config names
  - `src/device.py` — test `get_device()` returns a valid torch.device, `device_info()` returns expected keys
  - `src/evaluation/` — test metric functions with known inputs (accuracy=1.0 for perfect predictions)
  - `src/data/preprocess.py` — test `split_data()`, `encode_categoricals()`, `scale_features()`
- **REFACTOR:** Remove redundant tests, ensure each test adds coverage value.

**Acceptance criteria:**
- [ ] `pytest --cov=src` reports ≥ 80% overall coverage
- [ ] No single module below 60% coverage
- [ ] All tests pass
- [ ] `make test` exits 0

**Files:** Various test files (modify), possibly new `tests/test_config.py`, `tests/test_evaluation.py`
**Dependencies:** Tasks 2.1–2.5

---

### Checkpoint: Slice 2

- [ ] `make test` passes with coverage ≥ 80%
- [ ] `make docker-test` passes inside container
- [ ] No test takes longer than 10 seconds individually
- [ ] All existing functionality still works
- [ ] **Commit and verify `make lint` passes**

---

## Slice 3: MLflow Integration

**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`
**Delivers:** Models logged to MLflow, versioned in registry, visible at :5070

### Task 3.1: Add MLflow Dependency + Init/Fallback in BaseTrainer

**Description:** Add mlflow to dependencies and create MLflow initialization in BaseTrainer alongside existing W&B. Graceful fallback if MLflow server is unreachable.

**Scope:** M (2 files)

**Steps:**
- **RED:** Write `tests/test_mlflow_integration.py` with a test that creates a BaseTrainer subclass, runs training, and asserts `mlflow.active_run()` is not None. Use a temp MLflow tracking URI (`mlflow.set_tracking_uri(tmp_path / "mlruns")`). Test fails because BaseTrainer doesn't call MLflow.
- **GREEN:**
  1. Add `mlflow>=2.10` to `pyproject.toml` dependencies
  2. In `BaseTrainer.__init__()`, add `self._init_mlflow()` that:
     - Sets tracking URI from `MLFLOW_TRACKING_URI` env var (default: local `mlruns/`)
     - Calls `mlflow.start_run()` with run_name matching W&B pattern
     - Logs config as params
     - Sets `self.use_mlflow = True`
     - Catches connection errors → `self.use_mlflow = False` with warning
  3. In `log_metric()` / `log_metrics()`, also log to MLflow if `self.use_mlflow`
  4. In `finish()`, call `mlflow.end_run()` if active
- **REFACTOR:** Extract tracking setup into a private method for clarity.

**Acceptance criteria:**
- [ ] `mlflow` in pyproject.toml and installable via `uv sync`
- [ ] BaseTrainer logs to MLflow when server available
- [ ] BaseTrainer falls back gracefully when MLflow unreachable
- [ ] W&B logging unchanged (still optional)
- [ ] Test passes with local file-based tracking URI

**Files:** `pyproject.toml` (modify), `src/training/trainer.py` (modify), `tests/test_mlflow_integration.py` (new)
**Dependencies:** Task 2.1 (test fixtures)

---

### Task 3.2: MLflow Model Registry Integration

**Description:** After training, register the model in MLflow's model registry with versioning.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test asserting that after `trainer.run()`, the model is registered in MLflow with a version number. Test fails because `save_checkpoint()` doesn't register.
- **GREEN:**
  1. In `BaseTrainer.save_checkpoint()`, after saving local artifacts:
     - Log artifacts to MLflow run (`mlflow.log_artifacts(checkpoint_dir)`)
     - Register model: `mlflow.register_model(f"runs:/{run_id}/artifacts", model_name)` where `model_name` = problem name
     - Save `mlflow_run_id` and `mlflow_model_version` to metadata.json
  2. Handle registry errors gracefully (log warning, don't crash)
- **REFACTOR:** Ensure idempotent — re-running training creates new version, not duplicate.

**Acceptance criteria:**
- [ ] `metadata.json` includes `mlflow_run_id` after training
- [ ] Model registered in MLflow registry with version number
- [ ] Re-training increments version
- [ ] Test passes with temp tracking URI

**Files:** `src/training/trainer.py` (modify), `tests/test_mlflow_integration.py` (add tests)
**Dependencies:** Task 3.1

---

### Task 3.3: ModelPredictor MLflow Info + API Endpoint

**Description:** Update ModelPredictor to include MLflow version info, and update the `/models` API endpoint to surface it.

**Scope:** S (3 files)

**Steps:**
- **RED:** Add API test asserting `/models` response includes `mlflow_run_id` and `mlflow_model_version` keys for each model. Test fails because metadata.json doesn't have these fields yet (from old checkpoints).
- **GREEN:**
  1. `ModelPredictor.get_model_info()` already reads metadata.json — MLflow fields from Task 3.2 will appear automatically once models are retrained.
  2. Add a new endpoint `GET /models/registry` that queries MLflow for registered model versions (useful when MLflow server is running).
  3. Handle case where metadata.json is from pre-MLflow era (missing fields → null/unknown).
- **REFACTOR:** Ensure backward compatibility — old checkpoints without MLflow fields still work.

**Acceptance criteria:**
- [ ] `/models` includes MLflow version info (when available)
- [ ] Old checkpoints (pre-MLflow) don't break the endpoint
- [ ] Tests pass for both MLflow-enabled and legacy checkpoints

**Files:** `src/serving/predictor.py` (modify), `src/serving/api.py` (modify), `tests/test_api_endpoints.py` (add tests)
**Dependencies:** Task 3.2, Task 2.4

---

### Task 3.4: Docker-Compose MLflow Wiring Verification

**Description:** Verify the MLflow service in docker-compose works end-to-end with the training code. Ensure API and UI containers can reach MLflow.

**Scope:** S (1-2 files)

**Steps:**
- **RED:** Start docker compose, run training inside container, check MLflow UI for experiment. Currently training doesn't log to MLflow.
- **GREEN:**
  1. Add `MLFLOW_TRACKING_URI=http://mlflow:5000` to ml-api environment in docker-compose.yml
  2. Verify MLflow UI at http://localhost:5070 shows experiments after training
  3. Add `MLFLOW_TRACKING_URI` to `.env.example` with documentation
- **REFACTOR:** Add MLflow volume persistence (already exists as `mlflow_data`), verify data survives container restart.

**Acceptance criteria:**
- [ ] `docker compose up` → MLflow accessible at http://localhost:5070
- [ ] Training inside container logs to MLflow
- [ ] MLflow data persists across `docker compose down/up`
- [ ] `.env.example` documents MLFLOW_TRACKING_URI

**Files:** `docker-compose.yml` (modify), `.env.example` (modify)
**Dependencies:** Tasks 3.1–3.3, Task 1.3

---

### Checkpoint: Slice 3

- [ ] `make train` logs experiments to MLflow (local tracking)
- [ ] `docker compose up` → MLflow UI shows registered models at :5070
- [ ] `/models` API returns version info
- [ ] All tests pass including MLflow integration tests
- [ ] **Commit and verify `make lint` passes**

---

## Slice 4: Model Explainability

**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`
**Delivers:** SHAP values for tree models, gradient importance for autoencoder, visible in API + Gradio

### Task 4.1: SHAP Explainer Module for Tree Models

**Description:** Create `src/explainability/shap_explainer.py` that computes SHAP values for XGBoost (credit risk) and LightGBM (price prediction).

**Scope:** M (3 files)

**Steps:**
- **RED:** Write `tests/test_explainability.py`:
  - `test_shap_credit_risk_returns_feature_importances(credit_risk_model)` — asserts return has `feature_names`, `shap_values`, `feature_importances` keys with correct shapes.
  - `test_shap_price_returns_feature_importances(price_model)` — same pattern.
  - Tests fail because `src/explainability/` doesn't exist.
- **GREEN:**
  1. Add `shap>=0.44` to pyproject.toml
  2. Create `src/explainability/__init__.py`
  3. Create `src/explainability/shap_explainer.py`:
     - `explain_tree_model(model, X: pd.DataFrame, feature_names: list[str]) -> dict` — uses `shap.TreeExplainer(model)`, returns `{"feature_names": [...], "shap_values": np.array, "feature_importances": dict}` sorted by absolute mean SHAP value.
- **REFACTOR:** Handle edge cases (single-row input, missing features).

**Acceptance criteria:**
- [ ] SHAP values computed for XGBoost credit risk model
- [ ] SHAP values computed for LightGBM price model
- [ ] Return dict has `feature_names`, `shap_values`, `feature_importances`
- [ ] `feature_importances` sorted by absolute importance (descending)
- [ ] Tests pass with tiny model fixtures

**Files:** `pyproject.toml` (modify), `src/explainability/__init__.py` (new), `src/explainability/shap_explainer.py` (new), `tests/test_explainability.py` (new)
**Dependencies:** Task 2.1 (tiny model fixtures)

---

### Task 4.2: Gradient Explainer for Fraud Autoencoder

**Description:** Create `src/explainability/gradient_explainer.py` that computes gradient-based feature importance for the fraud autoencoder.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test `test_gradient_fraud_returns_feature_ranking(fraud_autoencoder)` to `tests/test_explainability.py`. Asserts return has `feature_names` and `feature_importances` keys. Fails because module doesn't exist.
- **GREEN:** Create `src/explainability/gradient_explainer.py`:
  - `explain_autoencoder(model: FraudAutoencoder, X: np.ndarray, feature_names: list[str], device: torch.device) -> dict`
  - Enable gradients on input tensor, compute reconstruction error, backprop, take absolute gradient mean → feature importance
  - Return `{"feature_names": [...], "feature_importances": dict}` sorted by importance
- **REFACTOR:** Ensure no GPU memory leak (detach tensors, no_grad where possible).

**Acceptance criteria:**
- [ ] Gradient-based importance computed for autoencoder
- [ ] Return dict has `feature_names` and `feature_importances`
- [ ] Importances are non-negative floats
- [ ] Works on CPU device (Docker compatibility)
- [ ] Tests pass

**Files:** `src/explainability/gradient_explainer.py` (new), `tests/test_explainability.py` (modify)
**Dependencies:** Task 4.1 (module structure), Task 2.1 (fraud autoencoder fixture)

---

### Task 4.3: Explainability API Endpoints

**Description:** Add `/explain/credit-risk`, `/explain/price`, and `/explain/fraud` endpoints to FastAPI.

**Scope:** M (3 files)

**Steps:**
- **RED:** Add tests to `tests/test_api_endpoints.py`:
  - `test_explain_credit_risk_returns_shap_values` — POST with loan data, assert response has `feature_importances`
  - `test_explain_price_returns_shap_values` — same for price
  - `test_explain_fraud_returns_gradient_importance` — same for fraud
  Tests fail because endpoints don't exist.
- **GREEN:**
  1. Add explain methods to `ModelPredictor`:
     - `explain_credit_risk(data: dict) -> dict` — loads model, runs SHAP
     - `explain_price(data: dict) -> dict` — loads model, runs SHAP
     - `explain_fraud(data: dict) -> dict` — loads model, runs gradient explainer
  2. Add endpoints to `api.py`:
     - `POST /explain/credit-risk` using existing `LoanApplication` schema
     - `POST /explain/price` using existing `Property` schema
     - `POST /explain/fraud` using existing `Transaction` schema
- **REFACTOR:** Cache explainer instances (SHAP TreeExplainer is expensive to create repeatedly).

**Acceptance criteria:**
- [ ] Three `/explain/*` endpoints return feature importances
- [ ] Response includes sorted `feature_importances` dict
- [ ] Uses same Pydantic models as prediction endpoints
- [ ] Tests pass with TestClient

**Files:** `src/serving/predictor.py` (modify), `src/serving/api.py` (modify), `tests/test_api_endpoints.py` (modify)
**Dependencies:** Tasks 4.1, 4.2, Task 2.4

---

### Task 4.4: Gradio UI Explainability Sections

**Description:** Add feature importance bar charts to the credit risk, price prediction, and fraud detection tabs in Gradio.

**Scope:** M (1 file, but complex UI)

**Steps:**
- **RED:** Manual test — open Gradio UI, make a prediction, no explanation shown. (UI changes are verified visually, not with automated tests.)
- **GREEN:** Modify `app/gradio_app.py`:
  1. Credit Risk tab: after prediction, call `predictor.explain_credit_risk()`, show Plotly horizontal bar chart of feature importances below prediction output.
  2. Price Prediction tab: same pattern.
  3. Fraud Detection tab: call `predictor.explain_fraud()`, show bar chart.
  4. Use `gr.Plot()` component for Plotly charts.
- **REFACTOR:** Extract chart creation into a helper function (same pattern for all 3).

**Acceptance criteria:**
- [ ] Credit risk tab shows feature importance bar chart after prediction
- [ ] Price tab shows feature importance bar chart after prediction
- [ ] Fraud tab shows feature importance bar chart after prediction
- [ ] Charts use Plotly horizontal bar, sorted by importance
- [ ] UI remains responsive (explainability computed on-demand, not on page load)

**Files:** `app/gradio_app.py` (modify)
**Dependencies:** Tasks 4.1–4.3

---

### Checkpoint: Slice 4

- [ ] `/explain/credit-risk` returns SHAP feature importances
- [ ] `/explain/price` returns SHAP feature importances
- [ ] `/explain/fraud` returns gradient-based feature importances
- [ ] Gradio UI shows bar charts after predictions
- [ ] All explainability tests pass
- [ ] **Commit and verify `make lint` passes**

---

## Slice 5: Structured Logging + Production Polish

**Skills:** `incremental-implementation`, `test-driven-development`
**Delivers:** JSON logs in Docker, request logging middleware, enriched health checks

### Task 5.1: Logging Configuration Module

**Description:** Create `src/logging_config.py` with a JSON formatter and setup function that switches between JSON (Docker) and human-readable (terminal) output.

**Scope:** S (2 files)

**Steps:**
- **RED:** Write `tests/test_logging.py`:
  - `test_json_formatter_produces_valid_json` — configure logging with JSON formatter, emit a log, parse the output as JSON, assert keys: `timestamp`, `level`, `message`, `logger`.
  - `test_text_formatter_produces_readable_output` — same but text format, assert no JSON.
  - `test_setup_logging_respects_env_var` — set `LOG_FORMAT=json`, call `setup_logging()`, verify JSON output.
  Tests fail because module doesn't exist.
- **GREEN:** Create `src/logging_config.py`:
  - `JsonFormatter(logging.Formatter)` — formats log records as JSON dicts
  - `setup_logging(log_format: str | None = None)` — reads `LOG_FORMAT` env var (default: "text"), configures root logger with appropriate formatter, sets level from `LOG_LEVEL` env var (default: "INFO")
- **REFACTOR:** Ensure formatter handles extra fields, exception info, and stack traces in JSON.

**Acceptance criteria:**
- [ ] `LOG_FORMAT=json` → JSON log lines with timestamp, level, message, logger
- [ ] `LOG_FORMAT=text` → human-readable format (similar to Rich but via stdlib)
- [ ] `LOG_LEVEL` env var controls verbosity
- [ ] Tests pass

**Files:** `src/logging_config.py` (new), `tests/test_logging.py` (new)
**Dependencies:** None

---

### Task 5.2: Migrate BaseTrainer from Rich to Logging

**Description:** Replace `console.print()` calls in BaseTrainer with `logger.info()` calls while keeping the same information content.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test `test_trainer_logs_pipeline_steps` that captures log output during training, asserts key messages appear: "Loading data", "Preprocessing", "Training model", "Evaluating", "Saving checkpoint".
- **GREEN:**
  1. In `trainer.py`, add `import logging` and `logger = logging.getLogger(__name__)`
  2. Call `setup_logging()` in `run()` method (once, at start)
  3. Replace each `console.print()` with `logger.info()`, preserving the message content
  4. Keep `console` import for backward compatibility (subclasses may use it) but deprecate
- **REFACTOR:** Remove Rich formatting tags (`[bold blue]`) from log messages — let the formatter handle styling.

**Acceptance criteria:**
- [ ] BaseTrainer uses `logger.info()` instead of `console.print()`
- [ ] Pipeline steps logged: data loading, preprocessing, training, evaluation, checkpoint saving
- [ ] Training time and metrics logged at INFO level
- [ ] Rich console not used in BaseTrainer (except subclass compatibility)
- [ ] All existing training tests still pass

**Files:** `src/training/trainer.py` (modify), `tests/test_training_pipeline.py` (add log assertions)
**Dependencies:** Task 5.1, Task 2.5

---

### Task 5.3: FastAPI Request Logging Middleware

**Description:** Add middleware to FastAPI that logs every request with method, path, status code, and latency.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test `test_api_request_logging` that makes a request via TestClient and captures log output, asserts a log entry with method, path, status_code, and latency_ms fields appears.
- **GREEN:** In `api.py`:
  1. Import `setup_logging` and call it at module level
  2. Add `@app.middleware("http")` that:
     - Records `time.time()` before `await call_next(request)`
     - Logs: `{"method": request.method, "path": request.url.path, "status_code": response.status_code, "latency_ms": round((end - start) * 1000, 1)}`
  3. Logs at INFO level for 2xx/3xx, WARNING for 4xx, ERROR for 5xx
- **REFACTOR:** Exclude health check from access logs (too noisy in production).

**Acceptance criteria:**
- [ ] Every API request logged with method, path, status_code, latency_ms
- [ ] Log level varies by status code (INFO/WARNING/ERROR)
- [ ] Health check excluded from access logs
- [ ] Tests verify middleware produces log entries

**Files:** `src/serving/api.py` (modify), `tests/test_logging.py` (add middleware tests)
**Dependencies:** Task 5.1, Task 2.4

---

### Task 5.4: Enriched Health Check

**Description:** Update `/health` to return model availability status — which models have checkpoints and are loadable.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test asserting `/health` response includes `models` key with per-model status. Currently it returns only `{"status": "ok"}`.
- **GREEN:** Update `/health` in `api.py`:
  ```python
  {
      "status": "ok",
      "models": {
          "credit_risk": {"available": true},
          "fraud_detection": {"available": true},
          "price_prediction": {"available": true},
          "demand_forecasting": {"available": false}
      }
  }
  ```
  Check for `checkpoints/{model}/metadata.json` existence.
- **REFACTOR:** Add uptime, version info if helpful.

**Acceptance criteria:**
- [ ] `/health` returns model availability per model
- [ ] Response includes `status` and `models` keys
- [ ] Missing checkpoints show `"available": false`
- [ ] Tests validate response structure

**Files:** `src/serving/api.py` (modify), `tests/test_api_endpoints.py` (modify)
**Dependencies:** Task 2.4

---

### Checkpoint: Slice 5

- [ ] `LOG_FORMAT=json make serve` → JSON log lines in terminal
- [ ] `docker compose logs ml-api` → JSON log entries
- [ ] Training logs show pipeline steps at INFO level
- [ ] API request/response logged with latency
- [ ] `/health` returns model availability status
- [ ] All tests pass, coverage ≥ 80%
- [ ] **Commit and verify `make lint` passes**

---

## Final Verification

After all 5 slices:

1. `docker compose up --build` → 3 services healthy within 90 seconds
2. `curl localhost:8070/health` → model availability status
3. `curl -X POST localhost:8070/predict/credit-risk` → valid prediction
4. `curl -X POST localhost:8070/explain/credit-risk` → SHAP values
5. Browse http://localhost:5070 → MLflow registered models
6. Browse http://localhost:3070 → Gradio with explainability charts
7. `make test` → ≥ 80% coverage, zero failures
8. `docker compose logs ml-api` → JSON log entries
9. `make lint` → zero warnings

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyTorch CPU-only Docker image too large (>3GB) | Slow builds, large disk usage | Use `--extra-index-url` for CPU-only wheels, multi-stage build |
| SHAP TreeExplainer slow on large models | API latency spike for /explain endpoints | Cache explainer instance, compute on small batch |
| MLflow version conflicts with other deps | Install failure | Pin compatible version, test in fresh venv |
| Existing tests break after conftest.py rewrite | False failures block progress | Run existing tests first after rewrite, fix before proceeding |
| Docker health checks flaky on cold start | CI failures | Set generous `start_period` (30s+), increase retries |
| xgboost/torch import ordering breaks in Docker | Segfault on macOS libomp | Already handled in conftest.py, verify in Docker too |

---
---

# Implementation Plan: Phase A.1 — Data Streaming Foundation

> **Spec:** `SPEC.md` §"Phase A.1 — Data Streaming Foundation"
> **Parent plan:** `~/.claude/plans/lexical-purring-nebula.md` §Phase A.1
> **Unlocks:** All slices A.2–A.9, B.1–B.4, C

## Overview

Break Phase A.1 into **11 tasks across 4 sub-phases**. Each task uses TDD (RED → GREEN → REFACTOR) and is sized ≤5 files. Foundation primitives first (deps, credentials, stream utils), then orchestration (modality dispatcher, adapter), then integration (trainer + CLI + MLflow), then verification (E2E + network + docs).

## Resolved Open Questions (from SPEC §A.1)

- **Q1 — Zillow → housing schema mapping**: Write `src/data/adapters/housing_adapter.py` that maps Kaggle `computingvictor/zillow-market-analysis-and-real-estate-sales-data` columns to our canonical housing schema (`square_feet, bedrooms, bathrooms, year_built, lot_size_sqft, garage_spaces, has_pool, neighborhood_tier, proximity_to_city_center, price`). Missing features imputed with column median; extra columns dropped. Adapter is declared in YAML via `data.adapter: "housing_adapter"`. **Decision:** adapter module, not inline trainer code, so other industries reuse the pattern.
- **Q2 — Checkpoint dir backward compatibility**: **Dual-write strategy.** When `--modality synthetic` runs, trainer writes to BOTH `checkpoints/price_prediction/` (legacy path) AND `checkpoints/price_prediction_synthetic/`. Other modalities write only to `checkpoints/price_prediction_<modality>/`. Predictor stays unchanged in A.1; future slice updates it to prefer recommended modality. **Decision:** zero-downtime migration; no predictor changes required in A.1.
- **Q3 — MLflow run hierarchy**: **Nested runs when `--modality all`, single run otherwise.** Parent run tagged `modality: all`, name `price_prediction_comparison_<timestamp>`. Three child runs nested via `mlflow.start_run(nested=True)` tagged `modality: synthetic|stream|mixed`. Single-modality invocations stay top-level (preserves existing behavior). W&B uses `group="price_prediction_comparison_<ts>"` instead of nesting.

## Dependency Graph

```
Task A.1.1: Deps + pytest marker
    │
    ├──→ Task A.1.2: Credentials loader
    │        │
    │        └──→ Task A.1.3: stream.py primitives
    │                │
    │                ├──→ Task A.1.4: modality.py dispatcher
    │                │        │
    │                │        ├──→ Task A.1.5: housing_adapter (Zillow schema)
    │                │        │
    │                │        └──→ Task A.1.6: Config schema extension
    │                │                 │
    │                │                 └──→ Task A.1.7: train_price migration + MLflow nesting
    │                │                          │
    │                │                          ├──→ Task A.1.8: Checkpoint alias dual-write
    │                │                          │
    │                │                          └──→ Task A.1.9: CLI --modality flag
    │                │                                   │
    │                │                                   └──→ Task A.1.10: E2E 3-modality integration test
    │                │                                            │
    │                │                                            └──→ Task A.1.11: @mark.network test + README
```

---

## Sub-Phase A.1.a: Foundation Primitives

**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`, `security-and-hardening`
**Delivers:** Reusable library for fetching real datasets

### Task A.1.1: Add Streaming Dependencies + pytest network Marker

**Description:** Install `kagglehub` (for Kaggle cache-download) and `datasets` (for HF streaming). Register a `network` marker in pytest config so integration tests can be skipped by default.

**Scope:** XS (2 files)

**Steps:**
- **RED:** N/A (dep config change — verify via `uv sync` succeeds)
- **GREEN:**
  - Add to `pyproject.toml` `[project.dependencies]`: `"kagglehub>=0.3"`, `"datasets>=2.18"`
  - Add to `[tool.pytest.ini_options]`:
    ```toml
    markers = [
        "network: tests that hit real external APIs (skipped by default)",
    ]
    ```
  - Update pytest `addopts` to include `-m "not network"` so default runs skip network tests
- **REFACTOR:** Verify `uv.lock` updates cleanly, no conflicts with existing deps

**Acceptance criteria:**
- [ ] `uv sync` succeeds with new deps on Apple Silicon
- [ ] `import kagglehub; import datasets` works inside the venv
- [ ] `pytest --collect-only` shows `[skipped]` for any existing `@pytest.mark.network` (none yet, but machinery is ready)
- [ ] Existing test suite still passes: `make test` stays at ≥88% coverage

**Files:** `pyproject.toml`, `uv.lock`
**Dependencies:** None

---

### Task A.1.2: Kaggle Credentials Loader

**Description:** Create `src/data/kaggle_credentials.py` that loads Kaggle API credentials from env vars first, falls back to `~/.kaggle/kaggle.json`. Raises a clear error if neither is available. This module is called before any `kagglehub` operation.

**Scope:** S (2 files)

**Steps:**
- **RED:** Write `tests/test_kaggle_credentials.py`:
  - `test_load_from_env_vars` — set KAGGLE_USERNAME/KAGGLE_KEY via monkeypatch, assert loader returns dict with those values
  - `test_load_from_kaggle_json` — monkeypatch `HOME` to tmp_path containing `.kaggle/kaggle.json`, assert loader reads it
  - `test_env_wins_over_file` — both present → env vars take precedence
  - `test_raises_when_missing` — neither present → raises `RuntimeError` with helpful message
- **GREEN:** Implement `load_kaggle_creds() -> dict[str, str]` with helpful error messages pointing to both env var and file setup
- **REFACTOR:** Add `ensure_kaggle_env()` helper that exports credentials into `os.environ` for kagglehub to pick up

**Acceptance criteria:**
- [ ] 4 tests pass, 100% coverage on `kaggle_credentials.py`
- [ ] Clear error message when creds missing: mentions both env var option and `~/.kaggle/kaggle.json` option
- [ ] Env vars take precedence over file when both exist
- [ ] No hardcoded paths — uses `Path.home()`

**Files:** `src/data/kaggle_credentials.py` (new), `tests/test_kaggle_credentials.py` (new)
**Dependencies:** Task A.1.1

---

### Task A.1.3: Stream Primitives (`src/data/stream.py`)

**Description:** Implement the three public utilities defined in SPEC: `kaggle_cached()`, `hf_stream()`, `iter_batches()`. All heavy imports (`kagglehub`, `datasets`) happen inside functions for clean mocking and fast module import.

**Scope:** M (2 files)

**Steps:**
- **RED:** Write `tests/test_streaming.py` with 4 mocked tests:
  - `test_kaggle_cached_returns_path` — monkeypatch `kagglehub.dataset_download` → returns tmp dir → assert Path returned
  - `test_kaggle_cached_with_filename` — same + filename param → returns `dir/filename`
  - `test_hf_stream_yields_rows` — monkeypatch `datasets.load_dataset` returning a fake iterable → assert iteration works
  - `test_iter_batches_chunks_correctly` — 2500 rows, batch_size=1000 → yields 3 DataFrames sized (1000, 1000, 500); also test DataFrame input path and iterator input path
- **GREEN:** Implement `src/data/stream.py` with module docstring, `from __future__ import annotations`, and typed signatures. `kaggle_cached` calls `ensure_kaggle_env()` from A.1.2 before downloading. `hf_stream` accepts optional `token` kwarg from `HF_TOKEN` env var. `iter_batches` handles both DataFrame and Iterator[dict].
- **REFACTOR:** Add module-level logger `logger = logging.getLogger(__name__)`; log the cache path at INFO level after download. Pylance-clean type hints.

**Acceptance criteria:**
- [ ] 4 tests pass, ≥90% coverage on `stream.py`
- [ ] No top-level imports of `kagglehub` or `datasets` (verify with grep)
- [ ] `kaggle_cached` returns `pathlib.Path`, not `str`
- [ ] Functions accept and propagate logging calls
- [ ] `make lint` passes

**Files:** `src/data/stream.py` (new), `tests/test_streaming.py` (new/extend)
**Dependencies:** Task A.1.2

---

### Checkpoint A.1.a — Foundation Primitives
- [ ] `make test` green, ≥88% overall coverage
- [ ] `make lint` zero warnings
- [ ] Commit: `"Phase A.1.a: streaming foundation — deps, creds loader, stream.py primitives"`

---

## Sub-Phase A.1.b: Modality Orchestration

**Skills:** `incremental-implementation`, `test-driven-development`, `api-and-interface-design`
**Delivers:** The 3-modality dispatcher + first adapter

### Task A.1.4: Modality Dispatcher (`src/data/modality.py`)

**Description:** Implement `load_for_modality(modality, synthetic_loader, stream_slug, stream_file, stream_adapter)` per SPEC. Handles the three modalities consistently and adds the `modality` feature column for `mixed`.

**Scope:** M (2 files)

**Steps:**
- **RED:** Add 4 tests to `tests/test_streaming.py`:
  - `test_load_for_modality_synthetic` — calls synthetic_loader, returns its output, never touches network
  - `test_load_for_modality_stream` — mocked kaggle_cached + mocked read_csv + mocked adapter → returns adapted DataFrame
  - `test_load_for_modality_mixed` — returns concat with `modality` column containing both "synthetic" and "stream" values, row count = synthetic + stream
  - `test_load_for_modality_invalid_raises` — `"invalid"` → raises ValueError listing valid modalities
- **GREEN:** Implement `load_for_modality()`. Signature exactly per SPEC. Return type: `pd.DataFrame` always. For `mixed`, concat with `ignore_index=True` and add `modality` column before concat so the column is correctly populated per-row.
- **REFACTOR:** Factor out common "read kaggle CSV + adapt" into private `_load_stream()` helper to avoid duplication between `stream` and `mixed` branches.

**Acceptance criteria:**
- [ ] 4 new tests pass, ≥95% coverage on `modality.py`
- [ ] `mixed` output DataFrame has a `modality` column with correct per-row values
- [ ] Unknown modality raises ValueError with list of valid values
- [ ] Stream adapter is optional (defaults to identity)

**Files:** `src/data/modality.py` (new), `tests/test_streaming.py` (extend)
**Dependencies:** Task A.1.3

---

### Task A.1.5: Housing Adapter (Zillow → canonical schema)

**Description:** Create `src/data/adapters/housing_adapter.py` (+ `__init__.py`) that maps Kaggle `computingvictor/zillow-market-analysis-and-real-estate-sales-data` columns to our canonical housing schema. Resolves Q1 from SPEC.

**Scope:** S (3 files)

**Steps:**
- **RED:** Write `tests/test_housing_adapter.py`:
  - `test_adapter_renames_zillow_columns` — feed a synthetic Zillow-shaped DataFrame, assert output has canonical column names
  - `test_adapter_imputes_missing_columns` — if `garage_spaces` is missing from input, output fills with median (or 0 if entirely absent)
  - `test_adapter_drops_extra_columns` — input has extra fluff columns → output only has canonical columns
  - `test_adapter_output_schema_matches_synthetic` — output columns are a subset or equal to `generate_housing.py` schema
- **GREEN:** Implement `housing_adapter(df) -> pd.DataFrame` with a COLUMN_MAP dict. Best-effort mapping:
  - `square_feet` ← `SquareFootage` or `living_area_sqft` or `sqft_living`
  - `bedrooms` ← `Bedrooms` or `beds`
  - `bathrooms` ← `Bathrooms` or `baths`
  - `year_built` ← `YearBuilt` or `year_built`
  - `lot_size_sqft` ← `LotSize` or `lot_size_sqft`
  - `garage_spaces` ← `GarageSpaces` (impute 0 if missing)
  - `has_pool` ← `HasPool`/`pool` → 0/1 (impute 0)
  - `neighborhood_tier` ← derive from `ZipCode` price quintile, or default to 3 (middle)
  - `proximity_to_city_center` ← `DistanceToCBD` (impute median)
  - `price` ← `SalePrice` or `Price`
- **REFACTOR:** Move COLUMN_MAP to a module constant so adapter is inspectable; add docstring listing the canonical schema.

**Acceptance criteria:**
- [ ] 4 tests pass, ≥90% coverage on `housing_adapter.py`
- [ ] Output DataFrame has exactly the 10 canonical columns (subset of housing.csv schema)
- [ ] No KeyError for missing optional fields — imputation strategy documented in docstring
- [ ] `make lint` passes

**Files:** `src/data/adapters/__init__.py` (new), `src/data/adapters/housing_adapter.py` (new), `tests/test_housing_adapter.py` (new)
**Dependencies:** Task A.1.4

---

### Checkpoint A.1.b — Modality Orchestration
- [ ] `make test` green, ≥88% overall coverage
- [ ] `make lint` zero warnings
- [ ] Commit: `"Phase A.1.b: modality dispatcher + Zillow housing adapter"`

---

## Sub-Phase A.1.c: Price Prediction Migration

**Skills:** `incremental-implementation`, `test-driven-development`, `deprecation-and-migration`
**Delivers:** price_prediction trains in all three modalities end-to-end

### Task A.1.6: Config Schema Extension

**Description:** Extend `configs/price_prediction.yaml` with `data.source`, `data.kaggle_slug`, `data.stream_file`, `data.adapter` keys. Default `data.source: synthetic` for backward compatibility.

**Scope:** S (2 files)

**Steps:**
- **RED:** Write `tests/test_config_loader.py` test:
  - `test_price_config_has_source_field` — `load_config("price_prediction")` returns dict with `data.source == "synthetic"`
  - `test_price_config_has_kaggle_slug` — value is `"computingvictor/zillow-market-analysis-and-real-estate-sales-data"`
  - `test_missing_source_defaults_to_synthetic` — craft a temp config without `source` key, confirm loader fills it
- **GREEN:** Update `configs/price_prediction.yaml`:
  ```yaml
  data:
    source: synthetic  # synthetic | stream | mixed
    raw_data_path: data/raw/housing.csv
    kaggle_slug: computingvictor/zillow-market-analysis-and-real-estate-sales-data
    stream_file: zillow_sales.csv  # actual filename TBD at build time
    adapter: housing_adapter  # module name in src/data/adapters/
    n_samples: 30000
    test_size: 0.2
    random_seed: 42
  ```
  Update `src/config.py` `load_config()` to backfill `source: "synthetic"` if missing.
- **REFACTOR:** Add a brief comment block in the YAML explaining the 3 modalities.

**Acceptance criteria:**
- [ ] 3 new tests pass
- [ ] All 4 existing configs load without error (backward-compat check)
- [ ] `make test` stays green
- [ ] `make lint` passes

**Files:** `configs/price_prediction.yaml` (modify), `src/config.py` (modify for backfill), `tests/test_config_loader.py` (add tests)
**Dependencies:** Task A.1.5

---

### Task A.1.7: Migrate `train_price.py` to Modality Dispatch + MLflow Nesting

**Description:** Refactor `PricePredictionTrainer.load_data()` to dispatch via `load_for_modality()`. Extend `BaseTrainer` to support optional `modality` parameter and nested MLflow runs. Resolves Q3.

**Scope:** M (3 files)

**Steps:**
- **RED:** Extend `tests/test_training_pipeline.py`:
  - `test_price_trainer_synthetic_modality` — instantiate with `modality="synthetic"`, mock data tiny, assert checkpoint written to `checkpoints/price_prediction_synthetic/`
  - `test_price_trainer_stream_modality` — mock kaggle_cached + adapter, assert checkpoint at `checkpoints/price_prediction_stream/`
  - `test_price_trainer_mixed_modality` — similar, assert `checkpoints/price_prediction_mixed/`
  - `test_mlflow_nested_run_tags_set` — mock mlflow, assert parent has `modality: all` tag and children have individual tags (only when parent context is active)
- **GREEN:**
  - Update `BaseTrainer.__init__` to accept `modality: str | None = None`. When set, `self.checkpoint_dir = checkpoints/<problem>_<modality>`. When None, use legacy `checkpoints/<problem>/` path.
  - Update `PricePredictionTrainer.load_data()`:
    ```python
    from src.data.modality import load_for_modality
    from src.data.adapters import housing_adapter
    return load_for_modality(
        modality=self.config["data"]["source"],
        synthetic_loader=lambda: pd.read_csv(self.config["data"]["raw_data_path"]),
        stream_slug=self.config["data"]["kaggle_slug"],
        stream_file=self.config["data"]["stream_file"],
        stream_adapter=housing_adapter,
    )
    ```
  - Update `BaseTrainer` MLflow init: accept `parent_run_id: str | None = None`. When passed, call `mlflow.start_run(nested=True)`. Tag each run with `modality`.
- **REFACTOR:** Extract the checkpoint dir logic into `BaseTrainer.get_checkpoint_dir()` method so other trainers reuse it.

**Acceptance criteria:**
- [ ] 4 new tests pass with tiny-data fixtures
- [ ] Existing `test_training_pipeline.py` still passes (synthetic-only default path intact)
- [ ] `checkpoints/price_prediction_synthetic/model.pkl` + `metadata.json` written when modality="synthetic"
- [ ] `metadata.json` includes `"modality"` key
- [ ] MLflow nested runs tag parent and children correctly

**Files:** `src/training/trainer.py` (extend BaseTrainer), `src/training/train_price.py` (migrate load_data), `tests/test_training_pipeline.py` (extend)
**Dependencies:** Task A.1.6

---

### Task A.1.8: Backward-Compatible Checkpoint Alias (Resolves Q2)

**Description:** When `--modality synthetic` runs, also mirror the checkpoint into `checkpoints/price_prediction/` (legacy path) so the existing `ModelPredictor` keeps working without any changes in this slice. Zero-downtime migration.

**Scope:** S (2 files)

**Steps:**
- **RED:** Add test `test_synthetic_modality_mirrors_legacy_path`:
  - Run trainer with `modality="synthetic"`, assert both `checkpoints/price_prediction_synthetic/model.pkl` AND `checkpoints/price_prediction/model.pkl` exist with identical bytes
  - Run with `modality="stream"`, assert `checkpoints/price_prediction/` is NOT touched (existing file preserved)
- **GREEN:** In `BaseTrainer.save_checkpoint()`, after saving to `_<modality>/`, if `modality == "synthetic"`, copy (via `shutil.copytree(..., dirs_exist_ok=True)`) into the legacy path.
- **REFACTOR:** Factor the mirror logic into `_mirror_to_legacy_path()` private method. Add a `deprecation_note` log entry: "Legacy checkpoint path will be removed in Phase A.9; predictor should migrate to _<modality>/ paths."

**Acceptance criteria:**
- [ ] New test passes
- [ ] `ModelPredictor` loads `checkpoints/price_prediction/model.pkl` without code changes (verify via existing API test)
- [ ] Non-synthetic modalities do not overwrite the legacy path
- [ ] Deprecation log message emitted on mirror

**Files:** `src/training/trainer.py` (extend), `tests/test_training_pipeline.py` (add test)
**Dependencies:** Task A.1.7

---

### Task A.1.9: CLI `--modality` Flag + Comparison Report

**Description:** Extend `scripts/train.py` with `--modality {synthetic,stream,mixed,all}` flag. When `all`, runs three sequential trainings under one MLflow parent run and prints a comparison table. Writes one row per modality to `results/price_prediction_metrics.csv`.

**Scope:** S (2 files)

**Steps:**
- **RED:** Write `tests/test_train_cli.py`:
  - `test_cli_modality_synthetic` — invoke main with `--model price --modality synthetic`, assert single checkpoint + single results row
  - `test_cli_modality_all` — invoke with `--modality all`, assert 3 checkpoints + 3 results rows, and a "recommended" log line identifies the best-R² variant
- **GREEN:** Update `scripts/train.py`:
  - Add `--modality` CLI arg with choices
  - When `"all"` and model supports modalities (check config has `data.source`), loop through `[synthetic, stream, mixed]`. Create parent MLflow run, pass `parent_run_id` to each trainer. Collect test metrics. After loop, print rich table; identify best R² (or RMSE) modality; write to `metadata.json`'s `recommended: bool` field.
  - For models without `data.source` in config (the other 3 existing models), gracefully ignore `--modality` and behave as before.
- **REFACTOR:** Factor the "choose recommended modality" logic into `scripts/_comparison.py` so future multi-modality models reuse it.

**Acceptance criteria:**
- [ ] 2 new tests pass
- [ ] `uv run python scripts/train.py --model price --modality all` completes all three in one invocation
- [ ] `results/price_prediction_metrics.csv` has 3 rows (one per modality)
- [ ] Best modality flagged in rich console output and written to that checkpoint's metadata.json
- [ ] `--modality` ignored cleanly for other existing models (no regression)

**Files:** `scripts/train.py` (extend), `tests/test_train_cli.py` (new)
**Dependencies:** Task A.1.8

---

### Checkpoint A.1.c — Price Prediction Migration
- [ ] All 3 modalities train end-to-end with mocked data
- [ ] Legacy checkpoint path still works (no predictor regression)
- [ ] MLflow parent/child run hierarchy visible in local mlruns/
- [ ] `make test` green, ≥88% overall coverage
- [ ] `make lint` zero warnings
- [ ] Commit: `"Phase A.1.c: price_prediction trains in 3 modalities, MLflow nested runs"`

---

## Sub-Phase A.1.d: End-to-End Verification + Docs

**Skills:** `test-driven-development`, `documentation-and-adrs`, `source-driven-development`
**Delivers:** Network test, README, and deployment verification

### Task A.1.10: End-to-End Mocked Integration Test

**Description:** A single high-value integration test that runs the full pipeline — `--modality all` on price_prediction — with all external calls (kagglehub, datasets, wandb, mlflow) mocked. Verifies the comparison CSV shape, MLflow nesting, and metadata correctness without touching the network.

**Scope:** M (2 files)

**Steps:**
- **RED:** Write `tests/test_phase_a1_e2e.py::test_price_modality_all_end_to_end`:
  - monkeypatch `kagglehub.dataset_download` → tmp dir with a tiny CSV matching Zillow schema
  - monkeypatch `mlflow.start_run` to a context manager that captures tags and params
  - invoke `scripts.train.main()` with `args=["--model", "price", "--modality", "all", "--no-wandb"]`
  - Assert: 3 rows in results CSV, 3 checkpoint dirs, legacy alias updated, best modality logged, `metadata.json` has `modality` + `recommended` keys
- **GREEN:** Make the test green (most logic already exists from A.1.7–A.1.9; this test reveals gaps)
- **REFACTOR:** Improve error messages in the pipeline where test failures exposed unclear diagnostics

**Acceptance criteria:**
- [ ] Test runs in <5 seconds (tiny-data fixtures)
- [ ] Zero real network calls (verify by disabling network via `pytest-socket` or equivalent if installed; else by monkeypatch assertion)
- [ ] Covers the main happy path end-to-end
- [ ] Overall coverage ≥88%

**Files:** `tests/test_phase_a1_e2e.py` (new), minor fixups to A.1.x files as needed
**Dependencies:** Task A.1.9

---

### Task A.1.11: Network Integration Test + README Update

**Description:** Write one `@pytest.mark.network` test that fetches a tiny real Kaggle dataset to prove the actual integration works. Update `README.md` with a "Streaming Data" section and `CLAUDE.md` with a note about `~/.cache/kagglehub/` cache location.

**Scope:** S (3 files)

**Steps:**
- **RED:** Write `tests/test_streaming.py::test_kaggle_cached_real_tiny_dataset` marked `@pytest.mark.network`:
  - Skip if `KAGGLE_USERNAME` or `KAGGLE_KEY` env vars unset
  - Use `uciml/iris` (tiny, stable) as the target slug — confirm it downloads to `~/.cache/kagglehub/` and returns a valid Path
  - Assert cached file is ≤200 KB
- **GREEN:** Run locally with real creds once to confirm it passes. Document the run in a PR note.
- **REFACTOR:** Update `README.md`:
  - Add "Streaming Datasets" subsection explaining the three modalities
  - Document Kaggle credential setup (env vars + `~/.kaggle/kaggle.json` options)
  - Note that real data lives in `~/.cache/kagglehub/` and is not committed
  Update `CLAUDE.md` "Conventions" with: "Real datasets stream from Kaggle/HF; cache at `~/.cache/kagglehub/`, never in `data/raw/`."

**Acceptance criteria:**
- [ ] Network test passes manually with real Kaggle creds, skipped in `make test`
- [ ] README has a clear "Streaming Datasets" section with three-modality explanation
- [ ] CLAUDE.md updated with streaming convention note
- [ ] `.env.example` gains `KAGGLE_USERNAME=`, `KAGGLE_KEY=`, `HF_TOKEN=` entries (may have been done in A.1.1; verify)

**Files:** `tests/test_streaming.py` (extend), `README.md` (modify), `CLAUDE.md` (modify), `.env.example` (modify if needed)
**Dependencies:** Task A.1.10

---

### Checkpoint A.1 — Phase Complete

- [ ] `make test` green with ≥88% overall coverage
- [ ] `make lint` zero warnings
- [ ] `uv run pytest -m network tests/test_streaming.py` passes with creds (manual verify once)
- [ ] `uv run python scripts/train.py --model price --modality all` produces 3 checkpoints + comparison CSV row on local machine (manual verify with real Zillow data once)
- [ ] `du -sh data/raw/` stays under 50 MB
- [ ] `du -sh ~/.cache/kagglehub/` confirms Kaggle data lives outside repo
- [ ] All 4 existing models still train and predict correctly (no regression)
- [ ] Legacy `checkpoints/price_prediction/` still loads in predictor without code changes
- [ ] Commit: `"Phase A.1 complete: streaming foundation + 3-modality price_prediction"`
- [ ] Tag: `v1.1.0-phase-a1` (optional)

---

## Phase A.1 Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Zillow column names differ from COLUMN_MAP assumptions | Stream modality fails on real data | Task A.1.5 uses best-effort fallback list per column; during manual `/build` of that task, inspect the actual Kaggle CSV and update the map before committing |
| Kaggle dataset gets taken down or renamed | stream modality breaks globally | `kaggle_slug` is in YAML, not code. Can swap to an alternative dataset without touching trainer code |
| `mlflow.start_run(nested=True)` semantics differ across versions | Nested runs invisible in MLflow UI | Pin `mlflow>=2.10` (already set); test against the installed version; fallback to non-nested runs with `group` tag if needed |
| `datasets` library changes streaming API | hf_stream breaks later slices | Pin `datasets>=2.18,<3.0`; wrap in thin adapter so we can swap libraries if needed |
| Checkpoint dual-write bloats disk over time | 30% more disk on every train | Acceptable for A.1; Phase A.9 retires the legacy path entirely |
| kagglehub cache location differs per OS | Windows/WSL users may see cache in weird paths | Document expected location in README; test manually on macOS + Linux; Windows deferred |

---

## Phase A.1 Verification (Manual, after all tasks)

1. Fresh clone → `uv sync --extra dev` → no errors
2. Set Kaggle creds → `uv run python scripts/train.py --model price --modality stream` → downloads Zillow data to `~/.cache/kagglehub/`, produces `checkpoints/price_prediction_stream/`
3. `uv run python scripts/train.py --model price --modality all` → 3 checkpoints, comparison CSV, best-modality logged
4. `curl localhost:8070/predict/price` → still works (backward compat via legacy path)
5. `make test` → 266+ tests pass, ≥88% coverage
6. `pytest -m network tests/test_streaming.py` → passes with creds
7. `du -sh data/raw/` → <50 MB
8. `git status` → `data/raw/` unchanged; only src/, tests/, configs/, docs/ touched
