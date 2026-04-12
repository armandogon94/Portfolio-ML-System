# Task List: Portfolio ML System — Production Hardening

## Slice 1: Docker Foundation + Dev Workflow
**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

- [ ] **1.1** Create .dockerignore + rename docker-compose.yml → docker-compose.prod.yml _(XS)_
- [ ] **1.2** Create Dockerfile.api — multi-stage, CPU PyTorch, non-root user _(S)_
- [ ] **1.3** Create Dockerfile.ui + self-contained docker-compose.yml (SQLite MLflow) _(M)_
- [ ] **1.4** Add Makefile docker targets (docker-build, docker-up, docker-down, docker-logs, docker-test) _(XS)_

**Checkpoint 1:** `docker compose up --build` → 3 healthy services, `curl :8070/health` → ok

---

## Slice 2: Test Infrastructure + 80% Coverage
**Skills:** `test-driven-development`, `incremental-implementation`

- [ ] **2.1** Configure pytest-cov + rewrite conftest.py with tiny model fixtures _(M)_
- [ ] **2.2** Parametrized data generator tests (4 generators × multiple sizes) _(S)_
- [ ] **2.3** Parametrized feature engineering tests (4 pipelines, derived value checks) _(S)_
- [ ] **2.4** API integration tests with TestClient (all endpoints) _(M)_
- [ ] **2.5** Training pipeline integration test (credit risk + fraud end-to-end) _(M)_
- [ ] **2.6** Coverage audit + gap filling to reach 80% _(M)_

**Checkpoint 2:** `make test` → ≥80% coverage, `make docker-test` passes

---

## Slice 3: MLflow Integration
**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

- [ ] **3.1** Add mlflow dep + MLflow init/fallback in BaseTrainer _(M)_
- [ ] **3.2** MLflow model registry (version on save_checkpoint) _(S)_
- [ ] **3.3** ModelPredictor MLflow info + /models endpoint update _(S)_
- [ ] **3.4** Docker-compose MLflow wiring verification _(S)_

**Checkpoint 3:** MLflow UI at :5070 shows experiments, `/models` returns version info

---

## Slice 4: Model Explainability
**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`

- [ ] **4.1** SHAP explainer module for XGBoost + LightGBM _(M)_
- [ ] **4.2** Gradient explainer for fraud autoencoder _(S)_
- [ ] **4.3** API /explain/* endpoints (credit-risk, price, fraud) _(M)_
- [ ] **4.4** Gradio UI explainability bar charts (3 tabs) _(M)_

**Checkpoint 4:** `/explain/credit-risk` returns SHAP values, Gradio shows charts

---

## Slice 5: Structured Logging + Production Polish
**Skills:** `incremental-implementation`, `test-driven-development`

- [ ] **5.1** Create logging_config.py (JSON/text formatter, env-driven) _(S)_
- [ ] **5.2** Migrate BaseTrainer from Rich to stdlib logging _(S)_
- [ ] **5.3** FastAPI request logging middleware _(S)_
- [ ] **5.4** Enriched /health endpoint (model availability status) _(S)_

**Checkpoint 5:** `docker compose logs ml-api` → JSON entries, `/health` → model status

---

## Final Verification

- [ ] `docker compose up --build` → 3 healthy services ≤90s
- [ ] All 4 `/predict/*` endpoints return valid JSON
- [ ] `/explain/credit-risk` + `/explain/price` return SHAP values
- [ ] MLflow UI at :5070 shows registered models
- [ ] Gradio at :3070 shows explainability charts
- [ ] `make test` → ≥80% coverage, 0 failures
- [ ] `docker compose logs ml-api` → JSON log entries
- [ ] `make lint` → 0 warnings
