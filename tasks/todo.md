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

- [x] **5.1** Create logging_config.py (JSON/text formatter, env-driven) _(S)_
- [x] **5.2** Migrate BaseTrainer from Rich to stdlib logging _(S)_
- [x] **5.3** FastAPI request logging middleware _(S)_
- [x] **5.4** Enriched /health endpoint (model availability status) _(S)_

**Checkpoint 5:** ✅ 266 tests, 88% coverage — JSON/text logging, trainer migration, request middleware, enriched /health — commit `5c0b170`

---

## Final Verification (Production Hardening)

- [x] `docker compose up --build` → 3 healthy services ≤90s
- [x] All 4 `/predict/*` endpoints return valid JSON
- [x] `/explain/credit-risk` + `/explain/price` return SHAP values
- [x] MLflow UI at :5070 shows registered models
- [x] Gradio at :3070 shows explainability charts
- [x] `make test` → ≥80% coverage, 0 failures
- [x] `docker compose logs ml-api` → JSON log entries
- [x] `make lint` → 0 warnings

---
---

# Task List: Phase A.1 — Data Streaming Foundation

> **Spec:** `SPEC.md` §"Phase A.1" · **Plan:** `tasks/plan.md` §"Phase A.1" · **Parent:** `~/.claude/plans/lexical-purring-nebula.md` §A.1

## Sub-Phase A.1.a: Foundation Primitives
**Skills:** `incremental-implementation`, `test-driven-development`, `source-driven-development`, `security-and-hardening`

- [x] **A.1.1** Add `kagglehub` + `datasets` deps, register `network` pytest marker, update default `addopts` _(XS)_ — commit `c3cb91c`
- [x] **A.1.2** `src/data/kaggle_credentials.py` — env-var + `~/.kaggle/kaggle.json` loader with tests _(S)_ — commit `ccccbd8`
- [x] **A.1.3** `src/data/stream.py` — `kaggle_cached()`, `hf_stream()`, `iter_batches()` with 9 mocked tests _(M)_ — commit `dbde94e`

**Checkpoint A.1.a:** foundation deps + primitives committed

---

## Sub-Phase A.1.b: Modality Orchestration
**Skills:** `incremental-implementation`, `test-driven-development`, `api-and-interface-design`

- [x] **A.1.4** `src/data/modality.py` — `load_for_modality()` dispatcher (synthetic/stream/mixed) + 7 tests _(M)_ — merge `bda7e9d` ← `7f631af`
- [x] **A.1.5** `src/data/adapters/housing_adapter.py` — Zillow → canonical schema mapping + 14 tests — **resolves Q1** _(S)_ — merge `50f4bf9` ← `7d28dfc`

**Checkpoint A.1.b:** ✅ dispatcher + adapter committed — 302 tests, 90% coverage

---

## Sub-Phase A.1.c: Price Prediction Migration
**Skills:** `incremental-implementation`, `test-driven-development`, `deprecation-and-migration`

- [x] **A.1.6** Extend `configs/price_prediction.yaml` with `data.source`, `data.kaggle_slug`, `data.stream_file`, `data.adapter`; backfill default in `src/config.py` _(S)_ — merge `ed64d25` ← `19d5df6`
- [x] **A.1.7** Migrate `train_price.py` to `load_for_modality()` + extend `BaseTrainer` with `modality` + nested MLflow runs — **resolves Q3** _(M)_ — commit `42c0095`
- [x] **A.1.8** Dual-write checkpoint: synthetic modality mirrors `checkpoints/price_prediction/` — **resolves Q2** _(S)_ — commit `9bed915`
- [x] **A.1.9** CLI `--modality {synthetic,stream,mixed,all}` in `scripts/train.py` + comparison report + "recommended" flag _(S)_ — commit `da4891d`

**Checkpoint A.1.c:** all 3 modalities train end-to-end, legacy predictor still works

---

## Sub-Phase A.1.d: End-to-End Verification + Docs
**Skills:** `test-driven-development`, `documentation-and-adrs`, `source-driven-development`

- [x] **A.1.10** End-to-end mocked integration test — `--modality all` happy path, zero network calls _(M)_ — commit `5842cf6`
- [x] **A.1.11** `@pytest.mark.network` test against real Kaggle + README + CLAUDE.md updates _(S)_ — commit `278cf75`

**Checkpoint A.1:** Phase A.1 complete — commit + optional tag `v1.1.0-phase-a1`

---

## Final Verification (Phase A.1)

- [ ] `make test` green with ≥88% coverage
- [ ] `make lint` zero warnings
- [ ] `pytest -m network` passes with real Kaggle creds (manual, once)
- [ ] `uv run python scripts/train.py --model price --modality all` produces 3 checkpoints + comparison CSV row
- [ ] `du -sh data/raw/` stays <50 MB
- [ ] `du -sh ~/.cache/kagglehub/` confirms real data outside repo
- [ ] All 4 existing models still train (no regression)
- [ ] Legacy `checkpoints/price_prediction/` still loads in ModelPredictor without code changes
