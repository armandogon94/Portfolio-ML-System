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

- [x] **A.2.8** `IndustryTile` + landing page with 6 tiles + 6 stub industry pages + shared `IndustryIndex` _(S)_ — 34 tests
- [x] **A.2.9** `ThemeToggle` (localStorage via next-themes) + `Nav` _(S)_ — 39 tests, smoke verified

**Checkpoint A.2.c:** ✅ full UI shell — landing + nav + dark mode all wired

---

## Sub-Phase A.2.d: Docker + Makefile
**Skills:** `source-driven-development`, `incremental-implementation`

- [x] **A.2.10** `Dockerfile.web` (prod multi-stage, standalone, non-root) + `docker-compose.yml` `ml-web:3071` service _(M)_ — 226 MB image, HTTP 200 verified
- [x] **A.2.11** `docker-compose.dev.yml` (HMR variant) + Makefile `web-*` targets + `web/README.md` _(S)_ — 8 new targets, compose validated

**Checkpoint A.2.d:** ✅ prod + dev docker workflows both wired; Makefile + docs cover all three run modes

---

## Sub-Phase A.2.e: Acceptance

- [x] **A.2.12** Final acceptance pass — all 15 SPEC success criteria verified _(XS)_ — commit `0037616`

**Checkpoint A.2:** ✅ Phase A.2 complete — 43 frontend tests, 97.43% components coverage, 323 Python tests green, backend untouched, Docker prod + dev wired, end-to-end proxy verified with live FastAPI

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


---

## Phase A.9: Retire Gradio + Dashboard Polish

**Skills:** `planning-and-task-breakdown`, `incremental-implementation`, `test-driven-development`, `deprecation-and-migration`, `documentation-and-adrs`

### Sub-Phase A.9.a — Foundation (parallel-safe)
- [ ] **A.9.1** Dynamic `/models` + `/health` — scan `checkpoints/*/metadata.json` instead of hardcoded lists _(S)_ — +3 tests
- [ ] **A.9.2** `web/lib/mlflow.ts` typed REST client + Zod schemas _(S)_ — +2 tests
- [ ] **A.9.3** Pre-existing Python lint debt cleanup (conftest, evaluate, run_all) _(XS)_

**Checkpoint A.9.a:** Python `make lint` clean · dynamic model scan live · MLflow client unit-tested

### Sub-Phase A.9.b — Legacy Model Ports (serial)
- [x] **A.9.4** Port `/fintech/fraud` (autoencoder + isolation forest) — commit `f17805b`, +2 tests
- [x] **A.9.5** Port `/real-estate/price` (LightGBM regressor) — commit `abb3daf`, +2 tests
- [x] **A.9.6** Port `/logistics/demand` (LSTM, 7-day forecast chart) — commit `6378f3a`, +2 tests

**Checkpoint A.9.b:** ✅ All 4 legacy Gradio tabs have Next.js equivalents · landing nav flows work · +6 web tests

### Sub-Phase A.9.c — Dashboard Build
- [x] **A.9.7** `MetricSparkline` + `ModelStatusBadge` components (TDD) — commit `5dc6a07`, +7 tests
- [x] **A.9.8** `DashboardTable` + `IndustrySummaryTile` components (TDD) — commit `10d2173`, +9 tests
- [x] **A.9.9** `/dashboard` Server Component + `getDashboardRows()` join — commit `6fe82e9`, +5 tests (route handlers skipped — `next.config.mjs` already proxies `/api/*`; ADR-noted)

**Checkpoint A.9.c:** ✅ `/dashboard` ships — 20-row table, 6 industry tiles, sparklines for ready models · +21 web tests

### Sub-Phase A.9.d — Gradio Retirement
- [x] **A.9.10** Parity snapshot test — Gradio ↔ Next.js output equivalence — commit `862e644`, +3 parity-marked tests
- [x] **A.9.11** Delete `app/gradio_app.py`, `Dockerfile.ui`, `ml-ui` compose service, `make ui` target + write `ADR-001-gradio-to-nextjs.md` — commit `f59ef2f`
- [x] **A.9.12** Final acceptance pass + tag `v1.4.0-phase-a-complete`

**Checkpoint A.9:** ✅ Phase A complete — 3 docker services, 10 Next.js demo routes, dashboard live, Gradio gone, ADR-001 committed, tag `v1.4.0-phase-a-complete` pushed

---

## Final Verification (Phase A.9) — ALL GREEN

- [x] `docker compose config --services` → exactly 3 (`mlflow`, `ml-api`, `ml-web`); no `ml-ui` reference
- [x] All 10 ready model pages prerender as static routes: credit-risk · fraud · price · demand · rental-price · dental/no-show · heart-disease · eta · churn · h1b-approval
- [x] `/dashboard` prerenders (4.04 kB) with `revalidate = 30` ISR
- [x] `make test` → 360 passed, 4 deselected (1 network + 3 parity), 90% coverage
- [x] `cd web && pnpm test` → 91 passed (was 57; +34 over Phase A.9)
- [x] `make lint` (Python) → All checks passed
- [x] `pnpm lint` + `pnpm typecheck` → 0 warnings, 0 errors
- [x] `pnpm build` → all 17 routes prerender; `/dashboard` listed
- [x] `git grep -i gradio` in active code → 0 matches (only ADR pointers remain)
- [x] `pytest -m parity` → 3/3 pass post-deletion (API contract preserved)
- [x] `git tag -l v1.4.*` → `v1.4.0-phase-a-complete` exists
- [x] ADR-001 at `docs/decisions/ADR-001-gradio-to-nextjs.md` is Accepted + complete (208 lines)

---

## Phase A — Total Delivered

| Slice | Title | Tag / Commit | Tests Δ |
|---|---|---|---|
| A.1 | Streaming foundation | tag `v1.1.0` | Python +N |
| A.2 | Next.js scaffold + credit-risk PoC | tag `v1.2.0-phase-a2-complete` | Web 0 → 43 |
| A.3–A.8 | 6 industry slices in parallel worktrees | tag `v1.3.0-phase-a-fanout` | Python 323 → 357, Web 43 → 57 |
| A.9.a | Foundation: dynamic /models + MLflow client + lint cleanup | `a671371` / `8b00195` / `bcfd6cc` | Python +3, Web +7 |
| A.9.b | Legacy ports: fraud, price, demand | `f17805b` / `abb3daf` / `6378f3a` | Web +6 |
| A.9.c | Dashboard: sparkline + badge + table + tile + page | `5dc6a07` / `10d2173` / `6fe82e9` | Web +21 |
| A.9.d | Gradio retirement: parity + atomic deletion + ADR | `862e644` / `f59ef2f` / `v1.4.0-phase-a-complete` | Python +3 parity-marked |

**Cumulative metrics (Phase A baseline → A.9.12 tag):**
- Python tests: 323 → **360** (+37, ≥90% coverage)
- Web tests: 0 → **91**
- Demo routes: 0 → **10 ready models + dashboard + 6 industry indexes**
- Docker services: `mlflow + ml-api + ml-ui` (Gradio) → `mlflow + ml-api + ml-web` (Next.js)
- Lint: 4 pre-existing warnings → **0**
- ADRs: 0 → **1**
