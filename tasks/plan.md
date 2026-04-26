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

## Parallelization Strategy (decided 2026-04-22)

Work will be dispatched in a **hybrid serial + parallel** pattern using the `Agent` tool with `isolation: "worktree"`:

| Phase | Mode | Notes |
|---|---|---|
| A.1.1 → A.1.3 | Serial | foundation primitives; serial for quality |
| **A.1.4 / A.1.5 / A.1.6** | **Parallel ×3 worktrees** | independent after A.1.3 merges |
| A.1.7 → A.1.11, A.2 | Serial | trainer + CLI + Next.js scaffold are cross-cutting |
| **A.3 → A.8** (industry slices) | **Parallel ×6 worktrees** | biggest parallel win; disjoint files per industry |
| **B.1 → B.4** | **Parallel ×4 worktrees** | also largely independent |
| Phase C | Serial | deploy, CI, docs |

**Skill:** `superpowers:dispatching-parallel-agents` invoked at each fan-out point. Merge conflicts on shared files (`src/serving/predictor.py`, `src/serving/api.py`, `README.md`) resolved by the main session after all worktree agents report back.

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

---
---

# Implementation Plan: Phase A.2 — Next.js Frontend Scaffolding

> **Spec:** `SPEC.md` §"Phase A.2 — Next.js Frontend Scaffolding"
> **Parent plan:** `~/.claude/plans/lexical-purring-nebula.md` §Phase A.2
> **Depends on:** Phase A.1 complete (FastAPI `/predict/*` + `/explain/*` endpoints serve the credit-risk model)
> **Unlocks:** Phases A.3–A.8 (6 industry slices) run in parallel worktrees atop this scaffold

## Overview

Break Phase A.2 into **12 tasks across 5 sub-phases**, delivering one end-to-end vertical slice (credit-risk PoC) on top of a production-grade Next.js + shadcn/ui + TanStack Query scaffold. Each task is sized ≤5 files; most are S/XS. A.2 is serial-only per the locked strategy — parallel fan-out starts at A.3.

## Resolved Open Questions (from SPEC §A.2)

- **Q-A.2-1 — Landing page copy:** Placeholder taglines written by me, each prefixed with `// TODO(copy)` for Armando to refine. Concrete copy:
  - Real Estate: "Price prediction, rental estimates, days-on-market"
  - Dental Clinics: "Cavity detection from X-rays, no-show prediction, treatment plans"
  - Healthcare: "Readmission risk, heart disease, diabetes onset, length-of-stay"
  - Fintech: "Credit risk, fraud detection, loan approval, customer churn"
  - Logistics: "Demand forecasting, delivery ETA, shipment damage risk"
  - Legal / Immigration: "H-1B approval, case duration, document classification"
- **Q-A.2-2 — Vercel deploy:** Deferred to Phase C. A.2 ensures Next.js `output: 'standalone'` works locally + in Docker; Vercel-specific glue (`vercel.json`, env var mapping, build hook) is Phase C.
- **Q-A.2-3 — Analytics / Sentry:** Deferred to Phase C. No Plausible, PostHog, or Sentry in A.2.

## Backend Constraint — Use Next.js Rewrites Instead of CORS

The SPEC locks in "No backend code modified — `git diff main src/` shows zero changes". FastAPI (`src/serving/api.py`) currently has no CORS middleware. Rather than adding `CORSMiddleware`, A.2 uses **Next.js rewrites** to proxy `/api/*` → FastAPI inside Next's server. The browser only talks to `:3071`; Next forwards to `:8070` server-side. Zero backend changes, no CORS preflight overhead.

`next.config.mjs`:
```javascript
const INTERNAL_API_URL = process.env.INTERNAL_API_URL ?? "http://localhost:8070";
export default {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${INTERNAL_API_URL}/:path*` }];
  },
};
```

`lib/api.ts` calls relative paths (`/api/predict/credit-risk`), not absolute URLs. `INTERNAL_API_URL` differs by environment: `http://localhost:8070` locally, `http://ml-api:8000` in Docker compose.

## Dependency Graph

```
Task A.2.1: Next.js init (pnpm, TS, Tailwind, ESLint, Prettier)
    │
    └──→ Task A.2.2: shadcn/ui install (primitives: button, card, input, label, form, slider, switch, toaster)
             │
             └──→ Task A.2.3: Root layout + providers (QueryClient, Theme) + placeholder page.tsx
                      │
                      └──→ Task A.2.4: Typed API client (lib/api.ts + lib/schemas.ts + next.config.mjs rewrites)
                               │
                               ├──→ Task A.2.5: ModelForm component + Vitest test
                               │        │
                               │        └──→ Task A.2.6: PredictionResult + ExplainabilityChart + Vitest tests
                               │                 │
                               │                 └──→ Task A.2.7: Credit risk page composition
                               │                          │
                               │                          └──→ Task A.2.12: Final acceptance pass
                               │
                               ├──→ Task A.2.8: IndustryTile + landing page (6 tiles)
                               │        │
                               │        └──→ Task A.2.9: ThemeToggle (localStorage) + Nav
                               │
                               ├──→ Task A.2.10: Dockerfile.web + docker-compose.yml ml-web service
                               │
                               └──→ Task A.2.11: docker-compose.dev.yml (HMR) + Makefile web-* targets
```

---

## Sub-Phase A.2.a: Scaffold (Tasks A.2.1–A.2.4)

**Skills:** `incremental-implementation`, `source-driven-development`, `frontend-ui-engineering`, `api-and-interface-design`
**Delivers:** `pnpm dev` starts a Next.js 14 server at `:3071` with providers wired, but no UI content yet.

### Task A.2.1: Initialize `web/` with Next.js 14 + TypeScript + Tailwind + pnpm

**Description:** Run `create-next-app` with the chosen flags, commit the scaffold. Configure `pnpm`, `tsconfig`, `eslint`, `prettier`. Set dev port to 3071.

**Scope:** M (5 files touched, mostly generated)

**Steps:**
- **RED:** N/A (scaffold task). Acceptance gate: `pnpm dev` starts a server.
- **GREEN:**
  - `cd <repo root>`; `pnpm create next-app@14 web --ts --tailwind --app --import-alias "@/*" --no-eslint --use-pnpm` (we add ESLint manually with our config)
  - `cd web`; install deps: `pnpm add @tanstack/react-query react-hook-form @hookform/resolvers zod recharts lucide-react`
  - Dev deps: `pnpm add -D vitest @vitejs/plugin-react @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @types/node eslint eslint-config-next prettier prettier-plugin-tailwindcss`
  - `web/package.json` scripts: `dev: "next dev -p 3071"`, `build: "next build"`, `start: "next start -p 3071"`, `lint: "next lint"`, `typecheck: "tsc --noEmit"`, `test: "vitest run"`, `test:watch: "vitest"`, `test:coverage: "vitest run --coverage"`
  - `.eslintrc.json` extends `next/core-web-vitals` + `prettier`
  - `.prettierrc.json` with `"plugins": ["prettier-plugin-tailwindcss"]`
  - `tsconfig.json` → `"strict": true`
  - `web/.gitignore` covering `node_modules`, `.next`, `coverage`, `.env*.local`
  - `web/.env.example` with `NEXT_PUBLIC_API_URL=http://localhost:8070` and `INTERNAL_API_URL=http://localhost:8070`
- **REFACTOR:** Delete the `create-next-app` boilerplate `app/page.tsx` content (will be replaced in A.2.3).

**Acceptance criteria:**
- [ ] `cd web && pnpm install` succeeds from clean clone
- [ ] `pnpm dev` serves http://localhost:3071 (not default 3000)
- [ ] `pnpm typecheck`, `pnpm lint`, `pnpm test` all run without errors (test suite empty but command works)
- [ ] `web/.gitignore` excludes `node_modules/`, `.next/`, `coverage/`

**Files:** `web/package.json`, `web/tsconfig.json`, `web/.eslintrc.json`, `web/.prettierrc.json`, `web/.gitignore`, `web/.env.example`
**Dependencies:** None

---

### Task A.2.2: Install shadcn/ui + Core Primitives

**Description:** Initialize shadcn/ui with `shadcn@latest init`, install the primitives A.2 needs.

**Scope:** M (generates `components/ui/*.tsx`, updates `tailwind.config.ts` + `app/globals.css`)

**Steps:**
- **RED:** N/A (install task). Acceptance gate: imports from `@/components/ui/*` work.
- **GREEN:**
  - `cd web`; `pnpm dlx shadcn@latest init --yes` with defaults: Default style, Slate base color, CSS variables for theme
  - Install primitives: `pnpm dlx shadcn@latest add button card input label form slider switch toast sonner`
  - Verify `components.json`, `lib/utils.ts` (with `cn()`), `tailwind.config.ts`, `app/globals.css` are generated/updated
- **REFACTOR:** Tidy `tailwind.config.ts` content globs to include `"./app/**/*.{ts,tsx}"`, `"./components/**/*.{ts,tsx}"`.

**Acceptance criteria:**
- [ ] `web/components/ui/` contains button.tsx, card.tsx, input.tsx, label.tsx, form.tsx, slider.tsx, switch.tsx, sonner.tsx
- [ ] `web/components.json` exists with chosen config
- [ ] `web/lib/utils.ts` exports `cn()`
- [ ] `pnpm typecheck` passes
- [ ] `pnpm lint` passes

**Files:** `web/components.json`, `web/components/ui/*.tsx` (generated), `web/lib/utils.ts`, `web/tailwind.config.ts`, `web/app/globals.css`
**Dependencies:** Task A.2.1

---

### Task A.2.3: Root Layout + Providers + Vitest Setup

**Description:** Wire `QueryClientProvider` and a theme provider (shadcn `next-themes`) into the root layout. Create `app/providers.tsx` as the client-side provider tree. Configure Vitest with jsdom.

**Scope:** M (5 files)

**Steps:**
- **RED:** Write `web/__tests__/smoke.test.tsx` — a trivial test that renders `<div>hi</div>` via `@testing-library/react` and asserts it's in the document. Fails because Vitest isn't configured yet.
- **GREEN:**
  - `pnpm add -D next-themes` (theme provider)
  - `web/vitest.config.ts`: jsdom env, include `__tests__/**/*.test.tsx`, setup file
  - `web/vitest.setup.ts`: imports `@testing-library/jest-dom`
  - `web/app/providers.tsx` (client): wraps children in `QueryClientProvider` (with a singleton client) + `ThemeProvider` (next-themes, attribute="class")
  - `web/app/layout.tsx`: server component root, imports Providers, sets `<html lang="en">`, applies Tailwind base
  - `web/lib/query-client.ts`: export singleton `QueryClient` with sane defaults (retry: 1, staleTime: 0)
- **REFACTOR:** Keep `layout.tsx` tiny — defer UI (nav etc.) to Task A.2.9.

**Acceptance criteria:**
- [ ] `pnpm test` passes (smoke test green)
- [ ] `pnpm dev` renders `<html lang="en">` with Tailwind base applied; browser DevTools shows no client-side errors
- [ ] QueryClient is a singleton (not re-created on every render) — verify by importing from two files and asserting identity

**Files:** `web/app/layout.tsx`, `web/app/providers.tsx`, `web/lib/query-client.ts`, `web/vitest.config.ts`, `web/vitest.setup.ts`, `web/__tests__/smoke.test.tsx`
**Dependencies:** Task A.2.2

---

### Task A.2.4: Typed API Client + Zod Schemas + next.config.mjs Rewrites

**Description:** Build `lib/api.ts` (typed client with `ApiError` + Zod response validation) and `lib/schemas.ts` (Zod input schemas per model). Configure Next.js `/api/:path*` rewrite to FastAPI.

**Scope:** M (3 files)

**Steps:**
- **RED:** Write `web/__tests__/lib/api.test.ts` — verifies `post()` throws `ApiError` on 4xx, calls schema.parse on success, targets a relative path. Uses `vi.mock` to stub `fetch`. Fails because `lib/api.ts` doesn't exist.
- **GREEN:**
  - `web/lib/schemas.ts`:
    - `CreditRiskInputSchema` (Zod): income, credit_score, debt_to_income, employment_years, loan_amount, num_accounts, num_late_payments (match FastAPI Pydantic model — check `src/serving/api.py` for exact field names)
    - Export inferred types: `export type CreditRiskInput = z.infer<typeof CreditRiskInputSchema>`
  - `web/lib/api.ts`:
    - `API_BASE_URL = "/api"` (relative — Next.js rewrites proxy)
    - `ApiError` class (status + body)
    - Private `post<T>(path, body, respSchema)` helper
    - Public `predictCreditRisk(input)` and `explainCreditRisk(input)` wired to `/predict/credit-risk` + `/explain/credit-risk`
    - Response schemas: `CreditRiskPredictionSchema`, `ExplanationSchema`
  - `web/next.config.mjs`: `output: "standalone"` + rewrites rule mapping `/api/:path*` → `${INTERNAL_API_URL}/:path*`
- **REFACTOR:** Split into per-model grouped sections with `// ─── Industry: <name> ───` comments so future slices can append cleanly.

**Acceptance criteria:**
- [ ] `pnpm test` passes (all A.2.4 tests green)
- [ ] `pnpm typecheck` clean
- [ ] `pnpm build` succeeds (validates next.config.mjs rewrites syntax)
- [ ] Start dev server + `curl http://localhost:3071/api/health` returns FastAPI's `/health` response (proxy works)

**Files:** `web/lib/api.ts`, `web/lib/schemas.ts`, `web/next.config.mjs`, `web/__tests__/lib/api.test.ts`
**Dependencies:** Task A.2.3

---

### Checkpoint A.2.a — Scaffold Complete
- [ ] `pnpm dev` starts without errors
- [ ] Providers wired (QueryClient + Theme)
- [ ] API client + Zod schemas in place, proxy working
- [ ] `pnpm typecheck`, `pnpm lint`, `pnpm test` all green
- [ ] Commit: `"Phase A.2.a: Next.js scaffold + shadcn/ui + TanStack Query + API client"`

---

## Sub-Phase A.2.b: Reusable Components + Credit Risk PoC (Tasks A.2.5–A.2.7)

**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`
**Delivers:** working credit-risk page exercising the full form → prediction → explainability loop.

### Task A.2.5: ModelForm Component + Vitest Test

**Description:** Build `ModelForm.tsx` — a reusable form wrapper that takes a Zod schema + field config and renders shadcn Inputs/Sliders, validates inline, calls `onSubmit` with parsed values.

**Scope:** S (2 files)

**Steps:**
- **RED:** Write `__tests__/components/ModelForm.test.tsx`:
  - `renders fields from config` — given 3 fields, shows 3 inputs with labels
  - `validates via Zod` — submit with invalid data shows inline error, onSubmit NOT called
  - `calls onSubmit with parsed values` — valid submission receives typed values
- **GREEN:** Implement `web/components/ModelForm.tsx` using `react-hook-form` + `@hookform/resolvers/zod` + shadcn Form primitives. Field config shape: `Array<{ name: string; label: string; type: "number" | "text" | "slider" | "select"; min?: number; max?: number; options?: string[] }>`.
- **REFACTOR:** Extract field-type switch into a `<ModelFormField />` private component so the map in ModelForm stays readable.

**Acceptance criteria:**
- [ ] 3 new tests pass, ≥85% coverage on ModelForm.tsx
- [ ] Form renders with fields, validates on submit, surfaces errors
- [ ] Typed: `<ModelForm<CreditRiskInput> />` works without `any`
- [ ] `pnpm lint` clean

**Files:** `web/components/ModelForm.tsx`, `web/__tests__/components/ModelForm.test.tsx`
**Dependencies:** Task A.2.4

---

### Task A.2.6: PredictionResult + ExplainabilityChart + Vitest Tests

**Description:** Build two result-display components. `PredictionResult` renders score/recommendation/confidence in a shadcn Card. `ExplainabilityChart` renders a Recharts horizontal bar chart of SHAP feature importances.

**Scope:** M (4 files)

**Steps:**
- **RED:** Write tests:
  - `__tests__/components/PredictionResult.test.tsx`:
    - Renders probability rounded to 2 dp
    - APPROVE is green, REVIEW is yellow, DECLINE is red (Tailwind class assertions)
    - Confidence displayed as percentage
  - `__tests__/components/ExplainabilityChart.test.tsx`:
    - Given 5 feature importances, renders 5 bars
    - Bars sorted by absolute value descending
    - Positive/negative signs produce different colors
- **GREEN:**
  - `web/components/PredictionResult.tsx`: accepts `{ probability, recommendation, confidence }` prop (typed via the Zod schema). Uses shadcn Card + Badge.
  - `web/components/ExplainabilityChart.tsx`: accepts `{ importances: Record<string, number>; topN?: number }`. Sorts by `|value|`, takes top N (default 10), renders Recharts BarChart.
- **REFACTOR:** Factor the recommendation→color map into a shared helper (e.g., `lib/ui-helpers.ts`).

**Acceptance criteria:**
- [ ] All new tests pass, ≥80% coverage on each component
- [ ] Components render in Storybook-free isolation (verify via test render)
- [ ] No `any` types

**Files:** `web/components/PredictionResult.tsx`, `web/components/ExplainabilityChart.tsx`, `web/__tests__/components/PredictionResult.test.tsx`, `web/__tests__/components/ExplainabilityChart.test.tsx`
**Dependencies:** Task A.2.5 (same reusable pattern)

---

### Task A.2.7: Credit-Risk Page Composition

**Description:** Build `app/fintech/credit-risk/page.tsx` — a client component that composes ModelForm + PredictionResult + ExplainabilityChart with TanStack Query mutations.

**Scope:** S (2 files)

**Steps:**
- **RED:** N/A (page-level compositions are easier to verify by manual smoke). Optional: a minimal render test that doesn't assert API behavior.
- **GREEN:**
  - `web/app/fintech/credit-risk/page.tsx` (client): two-column grid, left = ModelForm (with `CREDIT_RISK_FIELDS` config), right = result card + chart conditionally rendered based on mutation state
  - Uses `useMutation` from TanStack Query to call `predictCreditRisk` and `explainCreditRisk` (fire both on submit via `Promise.all`)
  - Loading state: shadcn Spinner / skeleton on the result column while pending
  - Error state: shadcn Sonner toast on `ApiError`
- **REFACTOR:** Extract `CREDIT_RISK_FIELDS` config to a co-located `fields.ts` so it can be unit-tested or reused.

**Acceptance criteria:**
- [ ] Manual smoke: fill form with plausible values, submit → see prediction + chart
- [ ] DevTools Network panel shows `POST /api/predict/credit-risk` + `POST /api/explain/credit-risk`
- [ ] Error case: backend down → toast shown, no crash

**Files:** `web/app/fintech/credit-risk/page.tsx`, `web/app/fintech/credit-risk/fields.ts`
**Dependencies:** Task A.2.6

---

### Checkpoint A.2.b — Credit Risk PoC Works End-to-End
- [ ] Submitting the credit-risk form produces a valid prediction + explainability chart
- [ ] DevTools Network confirms rewrites proxy works (no CORS errors)
- [ ] All Vitest tests green, ≥80% coverage on `web/components/`
- [ ] Commit: `"Phase A.2.b: credit-risk PoC — ModelForm + PredictionResult + ExplainabilityChart + page"`

---

## Sub-Phase A.2.c: Landing + Navigation (Tasks A.2.8–A.2.9)

**Skills:** `frontend-ui-engineering`, `incremental-implementation`
**Delivers:** landing page with 6 industry tiles, top navigation, working dark-mode toggle.

### Task A.2.8: IndustryTile + Landing Page

**Description:** Build the `IndustryTile` card and the landing `app/page.tsx` that renders 6 of them in a responsive grid.

**Scope:** S (2 files)

**Steps:**
- **RED:** Optional simple render test (`__tests__/components/IndustryTile.test.tsx`):
  - Renders icon, title, tagline, model count, CTA link
  - `href` prop routes to industry page
- **GREEN:**
  - `web/components/IndustryTile.tsx`: shadcn Card with lucide icon, title, `TODO(copy)` tagline, CTA "Try models →" linking to the industry index. Props: `{ href, title, tagline, icon: LucideIcon, modelCount }`.
  - `web/app/page.tsx` (server component): responsive grid (`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3`) with 6 tiles (RealEstate → `Home`, Dental → `Tooth`/fallback `Smile`, Healthcare → `Heart`, Fintech → `DollarSign`, Logistics → `Truck`, Legal → `Scale`). Each links to `/<industry>` (industry index is a stub for now — A.3–A.8 fill it in).
  - Create stub pages `app/<industry>/page.tsx` for each of the 6 so clicks don't 404 during A.2 (each shows a simple "Coming soon" + list of models per plan).
- **REFACTOR:** Extract the tile data array into a `lib/industries.ts` so A.3–A.8 agents can append to a single source of truth.

**Acceptance criteria:**
- [ ] Landing renders 6 tiles at `http://localhost:3071/`
- [ ] Each tile clickable to `/<industry>` stub page
- [ ] Responsive: 1 column at 375px, 2 cols at 768px, 3 cols at 1280px
- [ ] All TODO(copy) markers present for Armando to refine later

**Files:** `web/components/IndustryTile.tsx`, `web/app/page.tsx`, `web/lib/industries.ts`, `web/app/{real-estate,dental,healthcare,fintech,logistics,legal}/page.tsx` (6 stub pages)
**Dependencies:** Task A.2.4

---

### Task A.2.9: ThemeToggle + Nav

**Description:** Top nav with project title + dark mode toggle. Theme persists via `next-themes` (uses `localStorage` under the hood).

**Scope:** S (2 files)

**Steps:**
- **RED:** Test `__tests__/components/ThemeToggle.test.tsx`:
  - Clicking toggle flips `document.documentElement` class between "dark" and "light"
- **GREEN:**
  - `web/components/ThemeToggle.tsx` (client): uses `useTheme` from `next-themes`, renders shadcn Button with Sun/Moon icon
  - `web/components/Nav.tsx` (client): top bar with project title link to `/`, ThemeToggle on the right
  - Import Nav in `web/app/layout.tsx` so it appears on every page
- **REFACTOR:** Suppress hydration mismatch warning for theme-dependent rendering via `suppressHydrationWarning` on `<html>` (next-themes pattern).

**Acceptance criteria:**
- [ ] Nav visible on every page
- [ ] Dark mode toggle works and persists across reload (verify `localStorage.theme` in DevTools)
- [ ] No hydration mismatch warnings in console

**Files:** `web/components/ThemeToggle.tsx`, `web/components/Nav.tsx`, `web/app/layout.tsx` (modify to include Nav), `web/__tests__/components/ThemeToggle.test.tsx`
**Dependencies:** Task A.2.8

---

### Checkpoint A.2.c — UI Shell Ready
- [ ] Landing page renders 6 tiles
- [ ] Nav + ThemeToggle on every page, dark mode persists
- [ ] All tests green, ≥80% coverage on components
- [ ] Commit: `"Phase A.2.c: landing page with 6 industry tiles + nav + dark mode"`

---

## Sub-Phase A.2.d: Docker + Makefile (Tasks A.2.10–A.2.11)

**Skills:** `source-driven-development`, `incremental-implementation`
**Delivers:** `make docker-up` runs `ml-web:3071` alongside existing services; dev-compose variant supports HMR.

### Task A.2.10: Dockerfile.web (Prod) + docker-compose.yml Service

**Description:** Multi-stage `Dockerfile.web` using node:20-alpine, builds Next.js in standalone output, runs on port 3071 as non-root user. Add `ml-web` service to `docker-compose.yml` depending on `ml-api`.

**Scope:** M (2 files, 1 new)

**Steps:**
- **RED:** Shell test: `docker build -f Dockerfile.web -t ml-web-test .` must exit 0. `docker run --rm -p 3071:3071 ml-web-test` must serve a page on :3071 within 10s.
- **GREEN:**
  - `Dockerfile.web`:
    - Stage 1 (deps): `node:20-alpine`, install pnpm, `pnpm fetch`
    - Stage 2 (build): copy `web/`, run `pnpm install --offline`, `pnpm build` → produces `.next/standalone` + `.next/static`
    - Stage 3 (runner): `node:20-alpine`, non-root user `nextjs`, copy standalone output, expose 3071, CMD `node server.js`
    - `HEALTHCHECK` via `wget` or `curl` against `http://localhost:3071/`
  - `docker-compose.yml`: add `ml-web` service, port `3071:3071`, env `INTERNAL_API_URL=http://ml-api:8000`, `depends_on: ml-api`
- **REFACTOR:** Minimize final image size; ensure only standalone + static + public assets land in the runner stage.

**Acceptance criteria:**
- [ ] `docker build -f Dockerfile.web -t ml-web-test .` succeeds
- [ ] Image size < 500MB
- [ ] Runs as non-root user
- [ ] `make docker-up` brings up 4 services (mlflow, ml-api, ml-web, ml-ui legacy) healthy in <120s
- [ ] Inside compose network, ml-web successfully proxies to ml-api (verify via `docker compose logs ml-web` + submitting form)

**Files:** `Dockerfile.web` (new), `docker-compose.yml` (modify), `.dockerignore` (modify — ensure `web/node_modules` excluded)
**Dependencies:** Task A.2.9

---

### Task A.2.11: docker-compose.dev.yml (HMR) + Makefile web-* Targets

**Description:** Dev-mode compose override that mounts `web/` as a volume and runs `pnpm dev` inside the container for offline HMR. Add Makefile targets.

**Scope:** S (3 files)

**Steps:**
- **GREEN:**
  - `docker-compose.dev.yml` override for `ml-web`:
    - `command: pnpm dev`
    - `volumes: ["./web:/app", "/app/node_modules"]` (anonymous volume preserves container's installed deps)
    - `environment: [NODE_ENV=development]`
  - `Makefile` additions:
    ```make
    web-install:   cd web && pnpm install
    web-dev:       cd web && NEXT_PUBLIC_API_URL=http://localhost:8070 pnpm dev
    web-build:     cd web && pnpm build
    web-test:      cd web && pnpm test
    web-lint:      cd web && pnpm lint && pnpm typecheck
    docker-dev-up: docker compose -f docker-compose.yml -f docker-compose.dev.yml up web
    ```
  - Update `.env.example` root-level with the new `INTERNAL_API_URL` note
- **REFACTOR:** Document dev workflow in `web/README.md` — how to choose between `pnpm dev` (fastest), `make docker-dev-up` (HMR in container), and `make docker-up` (prod simulation).

**Acceptance criteria:**
- [ ] `make web-install` works from repo root
- [ ] `make web-dev` starts HMR at :3071
- [ ] `make web-test`, `make web-lint`, `make web-build` all succeed
- [ ] `make docker-dev-up` starts ml-web container with HMR + volume-mounted `web/`
- [ ] Editing a `.tsx` file in the dev container reflects in the browser within 2s

**Files:** `docker-compose.dev.yml` (new), `Makefile` (modify), `web/README.md` (new)
**Dependencies:** Task A.2.10

---

### Checkpoint A.2.d — Docker + Dev Workflow Ready
- [ ] Prod Docker image builds, runs, passes healthcheck
- [ ] Dev override mounts volume, HMR works
- [ ] Makefile targets all work
- [ ] Commit: `"Phase A.2.d: Dockerfile.web + compose integration + Makefile web targets"`

---

## Sub-Phase A.2.e: Acceptance (Task A.2.12)

### Task A.2.12: Final Acceptance Pass

**Description:** Walk through all 15 success criteria from SPEC §A.2. Fix any gaps found. Commit the phase checkpoint.

**Scope:** XS (verification only; minor fixups possible)

**Steps:**
1. Fresh clone simulation: `rm -rf web/node_modules web/.next`; `cd web && pnpm install && pnpm dev` → verify smoke
2. Run all SPEC success criteria manually (1 through 15)
3. Verify `git diff main src/ scripts/` shows zero changes (backend untouched)
4. Verify `app/gradio_app.py` still runs (parallel operation with A.2)
5. Run `pnpm test:coverage` and confirm ≥80% on `web/components/`
6. Run full Python suite: `make test` → 323+ tests pass, ≥90% coverage
7. `git log` final verification

**Acceptance criteria:**
- [ ] All 15 SPEC success criteria pass
- [ ] Python suite green
- [ ] Web `pnpm test:coverage` ≥80%
- [ ] No backend changes
- [ ] Commit: `"Phase A.2 complete: Next.js frontend + credit-risk PoC + Docker"`
- [ ] Optional tag: `v1.2.0-phase-a2`

**Files:** (verification only; may edit README.md or CLAUDE.md to document the new stack)
**Dependencies:** Task A.2.11

---

## Phase A.2 Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| pnpm not installed on dev machine | `pnpm install` fails | Document in `web/README.md`: `brew install pnpm` or `npm install -g pnpm`. Tasks A.2.1 does this once. |
| shadcn/ui CLI interactive prompts hang in CI / automation | Init blocks | Use `--yes` flag + pre-answered flags; document which defaults were chosen. |
| Next.js 14 vs 15 upstream — create-next-app@latest may install 15 | SPEC locked to 14 | Pin to `create-next-app@14` explicitly. Next 15 migration can be a separate task. |
| shadcn components importing `next/navigation` hooks break in Vitest (jsdom) | Component tests fail on hook errors | Add `vi.mock("next/navigation")` stubs in `vitest.setup.ts`. Documented in source. |
| Docker image size > 500MB | Violates acceptance criterion | Next.js `output: "standalone"` + Alpine base + multi-stage keep it under 250MB typically. If exceeded, audit with `docker history`. |
| CORS issue despite rewrites (misconfigured) | Browser blocks API calls | Verify with browser DevTools Network tab — Origin should be `http://localhost:3071` for every request, no preflight OPTIONS to :8070. |
| HMR slow in Docker dev mode (filesystem events) | Dev friction | Primary dev flow is `make web-dev` (native), not Docker dev. Docker dev is the offline fallback. |
| Legacy Gradio UI (`ml-ui:3070`) continues running in parallel | Potential confusion | Documented in SPEC: Gradio retires in A.9, not A.2. Dashboard page listed both URLs during transition. |

---

## Phase A.2 Verification (Manual Smoke, after A.2.12)

1. `cd web && pnpm install && pnpm dev` → http://localhost:3071 serves landing page with 6 industry tiles
2. Click **Fintech** → navigate to `/fintech/credit-risk` (or via Fintech stub)
3. Fill form: income=75000, credit_score=680, debt_to_income=0.35, etc. → Submit
4. Browser Network tab: `POST /api/predict/credit-risk` + `POST /api/explain/credit-risk`, both 200 OK
5. Right column renders: risk score card (APPROVE/REVIEW/DECLINE color-coded) + SHAP bar chart
6. Toggle dark mode — theme changes, reload → theme persists (localStorage check)
7. Resize browser to 375px width — layout stacks; 1280px — 3-column grid
8. `make docker-up` → 4 services healthy; browse `http://localhost:3071` → same UX
9. `make docker-dev-up` → HMR container; edit `app/page.tsx` → reflects in browser within 2s
10. `make web-test`, `make web-lint`, `make web-build` all succeed
11. `make test` (Python) → 323+ tests still pass, ≥90% coverage
12. `git diff main src/ scripts/` → empty (zero backend changes)

---

## Parallelization Opportunities

A.2 is **serial-only** per the locked strategy. No parallel worktree agents inside A.2.

Why serial: every task after A.2.4 (API client) shares mutable files — `app/layout.tsx`, `components/`, `tests/`, `docker-compose.yml`, `Makefile`. A parallel fan-out would produce merge conflicts on each.

The parallel fan-out resumes at **A.3** (Real Estate ×2), where each industry slice touches disjoint files (`web/app/<industry>/*`, `src/data/generate_<model>.py`, etc.) with only `web/lib/api.ts`, `web/lib/schemas.ts`, `web/lib/industries.ts`, `src/serving/predictor.py`, and `src/serving/api.py` as the shared append-points. Merge conflicts on those are trivially resolvable.

---

## Notes for Phase A.3–A.8 Builders (pattern reference)

When the parallel fan-out begins at A.3, each industry agent should:
1. Create `web/app/<industry>/<model>/page.tsx` by copy-pasting `web/app/fintech/credit-risk/page.tsx` and swapping:
   - Field config (`fields.ts`)
   - Zod schema import from `lib/schemas.ts`
   - API function import from `lib/api.ts`
2. Append to `web/lib/schemas.ts` a new grouped section for its industry
3. Append to `web/lib/api.ts` a new grouped section with `predict<Model>` + `explain<Model>`
4. Update the stub page at `web/app/<industry>/page.tsx` to list the new model
5. Add backend bits per the existing "Files to Touch Per New Model" pattern in this plan

A.2's acceptance gate ensures these appends are always clean — `api.ts` and `schemas.ts` are grouped by industry from day one.


---

# Phase A.9 Plan — Retire Gradio + Dashboard Polish

**Reference:** `SPEC.md` §Phase A.9 (appended 2026-04-23).
**Strategy:** Strangler pattern — build replacements first (legacy page ports + dashboard), verify parity, then delete Gradio. Each task is a vertical slice that leaves the system in a working state.

## Phase A.9 Dependency Graph

```
A.9.1  Dynamic /models + /health (backend)              [FOUNDATION]
A.9.2  MLflow REST client + Zod schemas (web/lib)       [FOUNDATION]
A.9.3  Lint debt cleanup (conftest, evaluate, run_all)  [FOUNDATION — unblocks make lint gate]
            │
            ▼
A.9.4  Port fraud  → /fintech/fraud                      [LEGACY PARITY]
A.9.5  Port price  → /real-estate/price                  [LEGACY PARITY]
A.9.6  Port demand → /logistics/demand                   [LEGACY PARITY]
            │
            ▼
A.9.7  MetricSparkline + ModelStatusBadge (TDD)          [DASHBOARD]
A.9.8  DashboardTable + IndustrySummaryTile (TDD)        [DASHBOARD]
A.9.9  /api route handlers + /dashboard page             [DASHBOARD]
            │
            ▼
A.9.10 Parity snapshot test: Gradio ↔ Next.js outputs   [RETIREMENT GATE]
A.9.11 Delete Gradio + write ADR-001                    [RETIREMENT]
A.9.12 Final acceptance + tag v1.4.0-phase-a-complete   [SHIP]
```

**Parallelization:** A.9.1 / A.9.2 / A.9.3 touch disjoint trees (backend Python / web TS / three Python files) — safe to run in parallel if desired. A.9.4 / A.9.5 / A.9.6 all append to `web/lib/industries.ts` + `web/lib/schemas.ts` + `web/lib/api.ts` but each goes into its own industry section; serial execution is simpler than worktree coordination for 3 small ports.

## Architecture Decisions

- **Dashboard is a Server Component with ISR** (`revalidate = 30`) — one round-trip to FastAPI + one to MLflow, cached 30s. No client-side data-fetching libs needed; avoids TanStack Query complexity for a read-only surface.
- **Dynamic model scan** — `get_model_info()` walks `checkpoints/*/metadata.json` instead of maintaining a hardcoded list. This future-proofs against B.1–B.4 model additions.
- **MLflow via REST API only** — no new Python package, no direct SQLite reads from the web app; keeps ml-web container slim and language-agnostic.
- **Strangler pattern for Gradio** — ports ship first, parity verified via snapshot test, then Gradio code deleted atomically in one commit. Git tag `v1.3.0-phase-a-fanout` is the rollback point.
- **ADR-001 supersedes the implicit Slice-1 "use Gradio" decision** — captures why we moved to Next.js, what alternatives were considered (Streamlit, Dash, keep Gradio), and the operational consequences.

---

## Sub-Phase A.9.a: Foundation (Tasks A.9.1–A.9.3)

### Task A.9.1: Dynamic `/models` + `/health` — Scan `checkpoints/` Directory

**Description:** Replace the hardcoded 5-model lists in `src/serving/predictor.py::get_model_info()` and `src/serving/api.py::_ALL_MODELS` with a dynamic scan of `checkpoints/*/metadata.json`. This means adding a new model no longer requires editing two Python files — just drop the checkpoint directory.

**Acceptance criteria:**
- [ ] `ModelPredictor.get_model_info()` returns one entry per `checkpoints/<problem>/metadata.json` that exists (no hardcoded list).
- [ ] `GET /health` reports `models: { <problem>: { available: bool } }` for every checkpoint dir under `checkpoints/` (also no hardcoded list).
- [ ] Backward-compat: response shape is unchanged for existing consumers (old legacy 5 keys still appear when trained).
- [ ] New tests: `test_get_models_scans_checkpoints_dir`, `test_health_reports_all_existing_checkpoints`, `test_health_reports_nothing_when_checkpoints_empty`.

**Verification:**
- `make test` → ≥360 passing (357 + ≥3 new)
- Manual: create a dummy `checkpoints/foo/metadata.json`, call `GET /models` → `foo` appears; delete it → disappears.

**Dependencies:** None

**Files likely touched:**
- `src/serving/predictor.py` (method body change)
- `src/serving/api.py` (remove `_ALL_MODELS`, inline dynamic scan in `/health`)
- `tests/test_serving.py` + `tests/test_logging.py::TestEnrichedHealth` (new tests)

**Scope:** S (3 files)

---

### Task A.9.2: MLflow REST Client + Zod Schemas

**Description:** Add `web/lib/mlflow.ts` — a thin typed client that calls MLflow's REST API (`/api/2.0/mlflow/experiments/search`, `/api/2.0/mlflow/runs/search`) from the Next.js server. Used by the dashboard sparkline feature. Responses are Zod-validated at the boundary.

**Acceptance criteria:**
- [ ] `getRunHistory(experimentName: string, metricKey: string): Promise<{ runs: Array<{ endTime: number; metric: number }> }>` — returns last 10 runs ordered by end_time desc.
- [ ] All MLflow responses validated with Zod; invalid shapes throw `MlflowError` with status + body.
- [ ] `MLFLOW_TRACKING_URI` env var controls the base URL; defaults to `http://mlflow:5000` (compose network) when unset.
- [ ] New tests: `web/__tests__/lib/mlflow.test.ts` — mocks `fetch`, verifies Zod parse + error handling, verifies the correct REST endpoint is called.

**Verification:**
- `cd web && pnpm test` → ≥59 passing (57 + 2 new)
- Manual (post-dashboard): dashboard renders sparklines for credit_risk (which has MLflow history).

**Dependencies:** None

**Files likely touched:**
- `web/lib/mlflow.ts` (new)
- `web/__tests__/lib/mlflow.test.ts` (new)
- `web/.env.local.example` (document `MLFLOW_TRACKING_URI`)

**Scope:** S (3 files)

---

### Task A.9.3: Pre-existing Python Lint Debt Cleanup

**Description:** Resolve the 4 pre-existing ruff warnings carried from earlier slices so `make lint` becomes a 0-warning gate going forward. Isolated from all other A.9 work; can ship standalone.

**Acceptance criteria:**
- [ ] `conftest.py:3` I001 — imports sorted.
- [ ] `scripts/evaluate.py:19` F841 — unused `args` removed (or prefixed with `_` if intentional).
- [ ] `scripts/run_all.py:29-30` E501 — lines broken to ≤100 chars.
- [ ] `make lint` → 0 errors, 0 warnings.
- [ ] Existing test suite still green (these are metadata-only fixes, no behavioral change).

**Verification:**
- `uv run ruff check .` → `All checks passed!`
- `make test` → 357 passing (unchanged)

**Dependencies:** None

**Files likely touched:**
- `conftest.py`
- `scripts/evaluate.py`
- `scripts/run_all.py`

**Scope:** XS (3 files)

---

### Checkpoint A.9.a — Foundation Ready

- [ ] `/models` + `/health` dynamically scan checkpoints
- [ ] MLflow REST client unit-tested
- [ ] `make lint` clean (0 warnings)
- [ ] Python: 357 → ≥360 tests passing
- [ ] Web: 57 → ≥59 tests passing

---

## Sub-Phase A.9.b: Legacy Model Ports (Tasks A.9.4–A.9.6)

Each task is the same vertical pattern used in A.3–A.8: new page + fields + test, plus 3 tiny appends to `industries.ts`, `schemas.ts`, `api.ts`. Reuses existing `ModelForm` + `PredictionResult` + `ExplainabilityChart`.

### Task A.9.4: Port Fraud Detection → `/fintech/fraud`

**Description:** Create a Next.js page for the fraud autoencoder + isolation-forest model that reaches parity with the Gradio fraud tab. Inputs: transaction_amount, merchant_category, hour_of_day, day_of_week, distance_from_home, is_online, card_age_days, num_transactions_last_hour, amount_vs_avg_ratio. Calls `POST /predict/fraud` + `POST /explain/fraud`. Result card shows anomaly score + fraud/legit recommendation.

**Acceptance criteria:**
- [ ] **RED test first:** `web/__tests__/app/fraud.test.tsx` — renders form, submits mock data, asserts result card shows probability + recommendation. Test fails before implementation.
- [ ] `FraudInputSchema` + `FRAUD_DEFAULTS` appended to `web/lib/schemas.ts` (Fintech section).
- [ ] `FraudPredictionSchema` + `predictFraud` + `explainFraud` appended to `web/lib/api.ts` (Fintech section).
- [ ] `industries.ts` — flip `fraud` to `ready: true`, point at `/fintech/fraud`.
- [ ] `web/app/fintech/fraud/{page.tsx,fields.ts}` — follows the 6 existing industry-page shapes exactly.
- [ ] Page renders in dev (`pnpm dev`), submission returns a prediction.

**Verification:**
- `pnpm test` → ≥61 tests passing (+2)
- `pnpm typecheck` clean, `pnpm lint` clean
- Manual: browse `/fintech/fraud`, submit defaults, confirm result + SHAP chart render

**Dependencies:** A.9.1 (for dashboard later; not strictly for this port), but none blocking.

**Files likely touched:**
- `web/app/fintech/fraud/page.tsx` + `fields.ts` (new)
- `web/__tests__/app/fraud.test.tsx` (new)
- `web/lib/schemas.ts`, `web/lib/api.ts`, `web/lib/industries.ts` (appends / flip)

**Scope:** S (6 files)

---

### Task A.9.5: Port Price Prediction → `/real-estate/price`

**Description:** Next.js port of the LightGBM price regressor. Inputs: square_feet, bedrooms, bathrooms, year_built, lot_size_sqft, garage_spaces, has_pool, neighborhood_tier, proximity_to_city_center. Calls `POST /predict/price` + `POST /explain/price`. Result card shows predicted price + confidence band.

**Acceptance criteria:**
- [ ] **RED test first:** `web/__tests__/app/price.test.tsx`.
- [ ] `PricePredictionInputSchema` + defaults appended to `web/lib/schemas.ts` (Real Estate section).
- [ ] `PricePredictionSchema` + API fns appended to `web/lib/api.ts`.
- [ ] `industries.ts` — flip `price` → `ready: true`, point at `/real-estate/price`.
- [ ] `web/app/real-estate/price/{page.tsx,fields.ts}` — mirrors rental-price page.

**Verification:**
- `pnpm test` → ≥63 tests passing (+2)
- Manual submission returns predicted price.

**Dependencies:** None

**Files likely touched:** same 6-file shape as A.9.4.

**Scope:** S (6 files)

---

### Task A.9.6: Port Demand Forecasting → `/logistics/demand`

**Description:** Next.js port of the PyTorch LSTM demand forecaster. Single dropdown input: `product` ∈ {electronics, apparel, groceries, furniture}. Calls `POST /predict/demand` — response is a 7-day forecast array. Result section renders a Recharts `<LineChart>` of the 7-day forecast (plus historical context if the API returns it).

**Acceptance criteria:**
- [ ] **RED test first:** `web/__tests__/app/demand.test.tsx` — verifies dropdown renders, submit returns forecast array, chart rendered.
- [ ] `DemandRequestSchema` + defaults in `web/lib/schemas.ts` (Logistics section).
- [ ] `DemandForecastSchema` (array of 7 numbers + metadata) + API fn in `web/lib/api.ts`.
- [ ] `industries.ts` — flip `demand` → `ready: true`, point at `/logistics/demand`.
- [ ] `web/app/logistics/demand/{page.tsx,fields.ts}` — fields.ts has a single `select` field; page renders the LineChart from the response.
- [ ] No `/explain/demand` call (the LSTM has no SHAP path) — hide ExplainabilityChart for this page.

**Verification:**
- `pnpm test` → ≥65 tests passing (+2)
- Manual: select "electronics", submit, see a 7-point forecast curve.

**Dependencies:** None

**Files likely touched:** same 6-file shape as A.9.4.

**Scope:** S (6 files)

**Risk:** ModelForm may not support single-dropdown layouts cleanly — if so, extend ModelForm to render a `select` field type (small addition; documented in A.2 SPEC as "extend renderControl").

---

### Checkpoint A.9.b — Legacy Ports Complete

- [ ] All 4 legacy models now have Next.js pages: credit-risk (existing) + fraud + price + demand.
- [ ] Python: 357 → ≥360 tests (from A.9.a, unchanged here)
- [ ] Web: 57 → ≥65 tests (+8 from ports)
- [ ] All 4 old Gradio tabs have a Next.js equivalent reachable via landing + industry pages.

---

## Sub-Phase A.9.c: Dashboard Build (Tasks A.9.7–A.9.9)

### Task A.9.7: `MetricSparkline` + `ModelStatusBadge` Components (TDD)

**Description:** Two small pure-Client components that the dashboard will use. `MetricSparkline` renders a Recharts `<LineChart>` styled as a 120×30 inline sparkline with no axes. `ModelStatusBadge` is a shadcn `<Badge>` variant showing `Ready` / `Training` / `Not Built`.

**Acceptance criteria:**
- [ ] **RED tests first** (`web/__tests__/components/MetricSparkline.test.tsx`, `ModelStatusBadge.test.tsx`) — written before components exist.
- [ ] `MetricSparkline` — props: `data: number[]`, optional `color: string`. Renders `<No history>` text if `data.length === 0`.
- [ ] `ModelStatusBadge` — prop: `status: "ready" | "training" | "not_built"`. Renders appropriate color + label.
- [ ] Both are pure Client Components (no data-fetching inside).
- [ ] ≥5 combined tests (empty, single point, 10 points, each status variant).

**Verification:**
- `pnpm test` → ≥70 tests (+5)
- `pnpm typecheck`, `pnpm lint` clean

**Dependencies:** None

**Files likely touched:**
- `web/components/MetricSparkline.tsx` (new)
- `web/components/ModelStatusBadge.tsx` (new)
- 2 test files (new)

**Scope:** S (4 files)

---

### Task A.9.8: `DashboardTable` + `IndustrySummaryTile` Components (TDD)

**Description:** Two larger Client Components that consume pre-fetched data. `DashboardTable` is a sortable, filterable table of all 20 model catalog entries joined with backend metadata. `IndustrySummaryTile` shows one industry's name, icon, ready-count/total, and the average key metric across ready models.

**Acceptance criteria:**
- [ ] **RED tests first:** renders 20 rows, filters by industry, sorts by metric column, shows empty state when rows empty.
- [ ] `DashboardTable` accepts a `rows: DashboardRow[]` prop (no fetching inside).
- [ ] Columns: industry · model · status badge · key metric (with unit) · last-trained ISO date · link (if ready).
- [ ] Sort by status (Ready first), sort by metric (numeric desc), sort by industry (alpha).
- [ ] `IndustrySummaryTile` — accepts `industry` + subset of `rows`; renders count + avg metric if ≥1 ready model.
- [ ] ≥7 combined tests.

**Verification:**
- `pnpm test` → ≥77 tests (+7)

**Dependencies:** A.9.7 (imports `ModelStatusBadge` + `MetricSparkline`)

**Files likely touched:**
- `web/components/DashboardTable.tsx` (new)
- `web/components/IndustrySummaryTile.tsx` (new)
- 2 test files (new)

**Scope:** M (4 files; DashboardTable holds the sort/filter state)

---

### Task A.9.9: Dashboard Page + Route Handlers + `lib/dashboard.ts`

**Description:** The `/dashboard` Server Component that joins 3 data sources into `DashboardRow[]`, plus the two Next.js Route Handlers (`/api/models`, `/api/mlflow-history`) it calls, plus the `lib/dashboard.ts` helper that orchestrates the fetches and maps them into display shape.

**Acceptance criteria:**
- [ ] **RED test first:** `web/__tests__/app/dashboard.test.tsx` — mocks the 3 data sources, asserts 20 rows render with correct status counts.
- [ ] `web/app/api/models/route.ts` — proxies `GET http://ml-api:8000/models`, validates with Zod.
- [ ] `web/app/api/mlflow-history/route.ts` — calls `lib/mlflow.ts::getRunHistory`, takes `?experiment=<name>&metric=<key>` query params.
- [ ] `web/lib/dashboard.ts::getDashboardRows()` — reads `INDUSTRIES` from `industries.ts`, joins with `/api/models` response, returns `DashboardRow[]`.
- [ ] `web/app/dashboard/page.tsx` — Server Component, `revalidate = 30`, renders 6 `IndustrySummaryTile`s + 1 `DashboardTable`.
- [ ] Header "Dashboard" link added to `web/components/Nav.tsx`.
- [ ] ≥3 additional tests (route handler Zod parsing, dashboard page smoke, empty API response).

**Verification:**
- `pnpm test` → ≥80 tests (+3)
- Manual: `make web-dev`, browse `/dashboard` → 6 tiles + 20-row table. Kill `ml-api`, reload → dashboard shows "Not Built" everywhere but doesn't crash.
- `pnpm build` — all routes prerender including new `/dashboard`.

**Dependencies:** A.9.1 (dynamic `/models`), A.9.2 (MLflow client), A.9.7 + A.9.8 (components)

**Files likely touched:**
- `web/app/dashboard/page.tsx` (new)
- `web/app/api/models/route.ts` (new)
- `web/app/api/mlflow-history/route.ts` (new)
- `web/lib/dashboard.ts` (new)
- `web/components/Nav.tsx` (add link)
- `web/__tests__/app/dashboard.test.tsx` (new)

**Scope:** M (6 files)

---

### Checkpoint A.9.c — Dashboard Ships

- [ ] `/dashboard` renders 20 models with live status, metrics, sparklines for ready models.
- [ ] Web: 57 → ≥80 tests (+23)
- [ ] Python: still ≥360 passing
- [ ] `pnpm build` succeeds with `/dashboard` listed as static or ISR

---

## Sub-Phase A.9.d: Gradio Retirement (Tasks A.9.10–A.9.12)

### Task A.9.10: Parity Snapshot Test — Gradio ↔ Next.js

**Description:** Before deleting Gradio, run one final parity check: submit default inputs via the FastAPI endpoints directly (both Gradio and Next.js call the same `/predict/*` routes, so parity at the API level proves parity at the UI level since both pages are pure wrappers). Capture JSON snapshots for fraud / price / demand responses and assert they're within numeric tolerance.

**Acceptance criteria:**
- [ ] New `tests/test_parity_gradio_nextjs.py` — hits `/predict/fraud`, `/predict/price`, `/predict/demand` with the Gradio default payloads, compares against committed snapshot JSON files in `tests/fixtures/gradio_parity/`.
- [ ] Snapshots captured at `v1.3.0-phase-a-fanout` checkpoint (i.e., before any A.9 changes). Use the current legacy pages' defaults.
- [ ] Test passes post-retirement — this is the gate that proves removing Gradio doesn't break model behavior.
- [ ] Marked `@pytest.mark.network` (requires live FastAPI) — excluded from `make test` by default; run explicitly before ship.

**Verification:**
- `uv run pytest -m network tests/test_parity_gradio_nextjs.py` → all pass

**Dependencies:** A.9.4, A.9.5, A.9.6

**Files likely touched:**
- `tests/test_parity_gradio_nextjs.py` (new)
- `tests/fixtures/gradio_parity/{fraud,price,demand}.json` (new snapshots)

**Scope:** S (4 files)

---

### Task A.9.11: Delete Gradio Code + Write ADR-001

**Description:** One atomic commit that retires Gradio completely. Deletes `app/gradio_app.py`, `Dockerfile.ui`, `ml-ui` service from `docker-compose.yml` (+ `.dev.yml`), the `ui:` Makefile target, and FRONTEND_PORT references. Updates README.md, CLAUDE.md, PORTS.md, `.env.example`. Commits `docs/decisions/ADR-001-gradio-to-nextjs.md`.

**Acceptance criteria:**
- [ ] `git grep -i gradio` returns matches only in ADR-001 + CHANGELOG / historical docs. Zero matches in `src/`, `app/` (which should no longer exist), `Makefile`, `docker-compose*.yml`, `web/`.
- [ ] `app/gradio_app.py` + `Dockerfile.ui` deleted; `app/` directory removed if empty.
- [ ] `make ui` target removed. `docker-compose.yml` no longer defines `ml-ui`. `.env.example` no longer defines `FRONTEND_PORT`.
- [ ] `PORTS.md` marks port 3070 as RELEASED (available for reuse).
- [ ] `README.md` — Gradio section removed; replaced with "Dashboard: http://localhost:3071/dashboard" + a `docker compose up` block showing 3 services.
- [ ] `CLAUDE.md` — Gradio references removed.
- [ ] `docs/decisions/ADR-001-gradio-to-nextjs.md` exists with: Status=Accepted, Date=2026-04-23, Context, Decision, Alternatives (Gradio retained, Streamlit, Dash, custom React), Consequences.
- [ ] `docker compose up --build` launches exactly 3 services, all healthy ≤120s.

**Verification:**
- `docker compose up --build -d` → `docker compose ps` shows 3 services all `healthy`
- `curl -s http://localhost:3071/dashboard` returns 200
- `make test` still 357+ green (no runtime behavior touched)

**Dependencies:** A.9.10 (parity proven)

**Files likely touched:**
- **Deleted:** `app/gradio_app.py`, `Dockerfile.ui`
- **Modified:** `docker-compose.yml`, `docker-compose.dev.yml`, `Makefile`, `README.md`, `CLAUDE.md`, `PORTS.md`, `.env.example`
- **New:** `docs/decisions/ADR-001-gradio-to-nextjs.md`, `docs/decisions/README.md` (index)

**Scope:** M (9–10 files — coordinated, but each edit is tiny)

---

### Task A.9.12: Final Acceptance + Tag `v1.4.0-phase-a-complete`

**Description:** Run the full SPEC §A.9 success criteria checklist end-to-end. Update `tasks/plan.md` + `tasks/todo.md` to mark A.9 complete. Create the release tag.

**Acceptance criteria:**
- [ ] All 11 SPEC §A.9 acceptance criteria tick ✅.
- [ ] `tasks/plan.md` + `tasks/todo.md` updated.
- [ ] `git tag -a v1.4.0-phase-a-complete` created with an annotated message summarizing Phase A (A.1→A.9).

**Verification:**
- Full acceptance run (compose up, /dashboard smoke, all 7+3 pages click-through, `make test`, `make lint`, `pnpm test`, `pnpm build`, `pnpm lint`)
- `git tag -l v1.4.*` shows the new tag

**Dependencies:** All previous A.9 tasks

**Files likely touched:** `tasks/plan.md`, `tasks/todo.md`

**Scope:** XS (2 files + 1 tag)

---

### Checkpoint A.9.d — Phase A Complete

- [ ] Gradio fully removed; 3 services in compose.
- [ ] Dashboard live at `/dashboard` with all 20 catalog entries.
- [ ] ADR-001 committed.
- [ ] Python: 357 → ≥360 tests. Web: 57 → ≥80 tests.
- [ ] `make lint` 0 warnings, `pnpm lint` 0 warnings.
- [ ] Tag `v1.4.0-phase-a-complete` pushed.

---

## Phase A.9 Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Gradio users break when tabs disappear mid-session | Low (solo dev demo) | Strangler pattern — Next.js ports live side-by-side with Gradio until A.9.11 delete step. |
| Demand-forecast Gradio chart used Plotly; Recharts port looks different | Low | Documented as acceptable regression in SPEC §Q-A.9-1. Recharts LineChart is visually comparable. |
| MLflow REST calls fail when MLflow container is down | Med | Dashboard treats missing history as "No history" empty-state — doesn't crash the page. Zod validation at boundary catches shape drift. |
| Dynamic checkpoint scan picks up stale / partial training dirs | Low | `metadata.json` existence is the gate — partial training runs don't write metadata. |
| Parity snapshot drifts when models are retrained | Low | Snapshots captured once at `v1.3.0-phase-a-fanout`; re-capture + commit if models retrained. Documented in test docstring. |
| Deleting `app/gradio_app.py` in same commit as compose changes produces huge diff | Low | Commit is intentionally atomic (one `chore(retire-gradio)` commit) — reviewers see the full migration in one place. Rollback = revert that single commit. |

---

## Open Questions

- **Q-A.9-P1 — Snapshot staleness:** if models are retrained in Phase B (adding modality comparisons for ported models), the parity snapshots in `tests/fixtures/gradio_parity/` will drift. Resolution: mark the parity test as `@pytest.mark.network` and refresh snapshots on retrain. Not a blocker.
- **Q-A.9-P2 — ADR numbering:** this is ADR-001 since no ADRs exist yet. If Phase B or C writes more ADRs, they'll be ADR-002+. `docs/decisions/README.md` will be the index.



---

# Phase A.9 — STATUS: COMPLETE

All 12 tasks shipped. Tagged at `v1.4.0-phase-a-complete` on 2026-04-26.

**Acceptance gate results (all green):**
- Docker: 3-service stack (`mlflow + ml-api + ml-web`) — `ml-ui` retired
- Routes: 10 ready model pages + `/dashboard` (with ISR) + 6 industry indexes — all prerender
- Python: 360 tests passing, 90% coverage; lint clean
- Web: 91 tests passing; typecheck + lint clean; build clean
- Parity: 3/3 snapshots match post-Gradio-deletion
- ADR-001: committed, status Accepted

Phase A is shipped. Phase B (industry depth — treatment-plan, diabetes,
LSTM length-of-stay, DistilBERT legal-doc classifier) and Phase C
(ship + Vercel deploy + GitHub Actions + case studies) are next.
