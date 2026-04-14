# Task List: Portfolio ML System — Production Hardening

## Slice 1: Docker Foundation + Dev Workflow
**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

- [x] **1.1** Create .dockerignore + rename docker-compose.yml → docker-compose.prod.yml _(XS)_
- [x] **1.2** Create Dockerfile.api — multi-stage, CPU PyTorch, non-root user _(S)_
- [x] **1.3** Create Dockerfile.ui + self-contained docker-compose.yml (SQLite MLflow) _(M)_
- [x] **1.4** Add Makefile docker targets (docker-build, docker-up, docker-down, docker-logs, docker-test) _(XS)_

**Checkpoint 1:** ✅ Committed `37c0db7`

---

## Slice 2: Test Infrastructure + 80% Coverage
**Skills:** `test-driven-development`, `incremental-implementation`

- [x] **2.1** Configure pytest-cov + rewrite conftest.py with tiny model fixtures _(M)_
- [x] **2.2** Parametrized data generator tests (4 generators × multiple sizes) _(S)_
- [x] **2.3** Parametrized feature engineering tests (4 pipelines, derived value checks) _(S)_
- [x] **2.4** API integration tests with TestClient (all endpoints) _(M)_
- [x] **2.5** Training pipeline integration test (credit risk + fraud end-to-end) _(M)_
- [x] **2.6** Coverage audit + gap filling to reach 80% _(M)_

**Checkpoint 2:** ✅ 208 tests, 88% coverage, 3.33s

---

## Slice 3: MLflow Integration
**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`

- [x] **3.1** Add mlflow dep + MLflow init/fallback in BaseTrainer _(M)_
- [x] **3.2** MLflow model registry (version on save_checkpoint) _(S)_
- [x] **3.3** ModelPredictor MLflow info + /models endpoint update _(S)_
- [x] **3.4** Docker-compose MLflow wiring verification _(S)_

**Checkpoint 3:** ✅ 218 tests, 88% coverage — MLflow init/fallback/registry/metadata all verified

---

## Slice 4: Model Explainability
**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`

- [x] **4.1** SHAP explainer module for XGBoost + LightGBM _(M)_
- [x] **4.2** Gradient explainer for fraud autoencoder _(S)_
- [x] **4.3** API /explain/* endpoints (credit-risk, price, fraud) _(M)_
- [x] **4.4** Gradio UI explainability bar charts (3 tabs) _(M)_

**Checkpoint 4:** ✅ 243 tests, 88% coverage — SHAP + gradient explainers, /explain/* API, Gradio charts — commits `5dbfca9`, `26cb631`

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
