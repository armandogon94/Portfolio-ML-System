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

- [x] `make test` green with ≥88% coverage (323 tests, 90%)
- [x] `make lint` zero warnings
- [ ] `pytest -m network` passes with real Kaggle creds (manual, once — awaits user)
- [ ] `uv run python scripts/train.py --model price --modality all` produces 3 checkpoints + comparison CSV row (manual, needs creds)
- [x] `du -sh data/raw/` stays <50 MB (unchanged)
- [x] `du -sh ~/.cache/kagglehub/` confirms real data outside repo (design-level verified)
- [x] All 4 existing models still train (no regression)
- [x] Legacy `checkpoints/price_prediction/` still loads in ModelPredictor without code changes

---
---

# Task List: Phase A.2 — Next.js Frontend Scaffolding

> **Spec:** `SPEC.md` §"Phase A.2" · **Plan:** `tasks/plan.md` §"Phase A.2"
> **Serial-only** (parallel fan-out resumes at A.3)

## Sub-Phase A.2.a: Scaffold
**Skills:** `incremental-implementation`, `source-driven-development`, `frontend-ui-engineering`, `api-and-interface-design`

- [x] **A.2.1** Initialize `web/` with Next.js 14 + TypeScript + Tailwind + pnpm (port 3071) _(M)_ — commit `bcff1f1`
- [x] **A.2.2** Install shadcn/ui + core primitives (button, card, input, label, form, slider, switch, sonner) _(M)_ — commit `3e96622`
- [x] **A.2.3** Root layout + Providers (QueryClient + Theme) + Vitest setup _(M)_ — commit `8b05f05`
- [x] **A.2.4** Typed API client (`lib/api.ts`) + Zod schemas (`lib/schemas.ts`) + `next.config.mjs` rewrites proxy _(M)_

**Checkpoint A.2.a:** ✅ scaffold ready — proxy verified end-to-end, 9 tests passing

---

## Sub-Phase A.2.b: Credit Risk PoC
**Skills:** `incremental-implementation`, `test-driven-development`, `frontend-ui-engineering`

- [x] **A.2.5** `ModelForm` component + Vitest test _(S)_ — commit `243c500`
- [x] **A.2.6** `PredictionResult` + `ExplainabilityChart` + Vitest tests _(M)_ — commit `f0dee47`
- [x] **A.2.7** Credit-risk page (`app/fintech/credit-risk/page.tsx`) composition _(S)_ — commit landed; 29/29 tests

**Checkpoint A.2.b:** ✅ credit-risk form → prediction → SHAP chart wired end-to-end

---

## Sub-Phase A.2.c: Landing + Navigation
**Skills:** `frontend-ui-engineering`, `incremental-implementation`

- [ ] **A.2.8** `IndustryTile` + landing page with 6 tiles + 6 stub industry pages _(S)_
- [ ] **A.2.9** `ThemeToggle` (localStorage via next-themes) + `Nav` _(S)_

**Checkpoint A.2.c:** full UI shell — landing, nav, dark mode

---

## Sub-Phase A.2.d: Docker + Makefile
**Skills:** `source-driven-development`, `incremental-implementation`

- [ ] **A.2.10** `Dockerfile.web` (prod multi-stage, standalone, non-root) + `docker-compose.yml` `ml-web:3071` service _(M)_
- [ ] **A.2.11** `docker-compose.dev.yml` (HMR variant) + Makefile `web-*` targets + `web/README.md` _(S)_

**Checkpoint A.2.d:** `make docker-up` runs ml-web alongside existing services; dev override supports HMR

---

## Sub-Phase A.2.e: Acceptance

- [ ] **A.2.12** Final acceptance pass — verify all 15 SPEC success criteria + commit phase tag _(XS)_

**Checkpoint A.2:** Phase A.2 complete — commit + optional tag `v1.2.0-phase-a2`

---

## Final Verification (Phase A.2)

- [ ] Fresh clone → `cd web && pnpm install && pnpm dev` → :3071 renders landing with 6 tiles
- [ ] Credit risk form → submission → prediction + SHAP chart rendered
- [ ] Dark mode toggle works + persists via localStorage
- [ ] Responsive: 375px / 768px / 1280px all render correctly
- [ ] `pnpm typecheck` clean, `pnpm lint` clean, `pnpm test` ≥80% coverage on components
- [ ] `make docker-up` → 4 services healthy including `ml-web:3071`
- [ ] `make docker-dev-up` → HMR container running
- [ ] Backend unchanged: `git diff main src/ scripts/` empty
- [ ] Python suite still green: `make test` → 323+ tests, ≥90% coverage
- [ ] Gradio (`ml-ui:3070`) still runs in parallel (retires in A.9)
