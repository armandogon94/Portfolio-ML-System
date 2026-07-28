# ADR-001: Replace Gradio with Next.js + shadcn/ui for the demo UI

## Status/Amendment: 2026-07-25

**Accepted; historical references amended.** The decision and its original
reasoning remain below unchanged. `SPEC.md`, `tasks/plan.md`, the parity test,
and `tests/fixtures/gradio_parity/` were later deleted when the repository
narrowed to three fintech problems; current architecture and verification live
in `docs/architecture.md`, `docs/PROGRESS.md`, and
`scripts/verify_fresh_clone.sh`.

The original verification claim of “360+ tests” is stale. The narrowed suite
was **181 passing tests at commit `8d9e9bc`**, measured by running
`./.venv/bin/pytest -p no:cacheprovider --no-cov -m "not network"` at that
commit. Subsequent work has added tests, so `docs/PROGRESS.md` records the latest
command output instead of treating 181 as a permanent count.

## Status

**Accepted**: implemented in Phase A.9 (slices A.9.2 through A.9.11).
Supersedes the implicit Slice 1 decision to use Gradio as the demo UI.

## Date

2026-04-25

## Context

The original Slice 1 deliverable (committed at v1.0.0) used Gradio as
the demo interface: a Python-native UI library that ships with FastAPI
on the same Docker image. That choice was right for the prototype phase:
Gradio let one Python file render five tabs of forms + result cards
with no frontend toolchain, and shipped a working UI in an afternoon.

By the time Phase A's industry expansion was scoped (20 models across
6 industries: Real Estate, Dental, Healthcare, Fintech, Logistics,
Legal/Immigration), Gradio had become the project's primary
constraint:

1. **Visual ceiling.** Every Gradio app looks like a research
   prototype. The portfolio is being shown to prospective clients in
   each of the 6 industries; "looks like a product, not a notebook"
   is itself a key signal of senior engineering.

2. **Layout rigidity.** Per-industry pages need bespoke result cards
   (4-tier risk badges, 7-day forecast charts, SHAP bar charts), a
   home-grown landing page with industry tiles, an aggregate live
   `/dashboard` with sortable tables and sparklines. Gradio's `Blocks`
   API can technically do most of this but every step fights the
   default styling.

3. **Per-industry deep links.** The plan needs URLs like
   `/fintech/credit-risk` and `/dental/no-show` so a recruiter or
   client can be sent straight to one model. Gradio's tabs all live at
   the same URL.

4. **Dark mode + responsive layout.** Both essentially free in
   Tailwind + shadcn; both manual + half-broken in Gradio.

5. **Public deploy path.** A static-rendered Next.js bundle deploys
   to Vercel in one click. A Gradio app needs a Python server, which
   needs a non-trivial host. Phase C ships publicly; the stack matters.

6. **Type-safe inputs.** Zod-validated forms catch shape drift at the
   boundary. Gradio inputs are dynamically typed: every form drift
   is a 422 from FastAPI.

The team had to choose between piling Gradio workarounds for each
industry or paying the one-time cost of moving to a real frontend.

## Decision

**Replace Gradio with a Next.js 14 (App Router) + TypeScript + Tailwind
+ shadcn/ui app, deployed as a third Docker service (`ml-web`) behind
the same FastAPI backend.**

The migration was executed via the **strangler pattern**:

- **A.9.2–A.9.6**: built Next.js ports for every legacy Gradio tab
  (credit-risk, fraud, price, demand-forecast). Both UIs ran in
  parallel against the same FastAPI endpoints.
- **A.9.7–A.9.9**: built the new `/dashboard` page that didn't exist
  in Gradio at all (live status table + per-industry summary tiles
  + MLflow training-run sparklines).
- **A.9.10**: captured JSON parity snapshots of `/predict/{fraud,
  price,demand}` for the legacy default inputs. Both UIs are pure
  wrappers around these endpoints, so byte-for-byte parity at the
  API level is byte-for-byte parity at the UI level.
- **A.9.11** (this ADR): deleted `app/gradio_app.py`, `Dockerfile.ui`,
  the `ml-ui` compose service, the `make ui` Makefile target, and the
  Gradio dependency from `pyproject.toml`: all in one atomic commit.

Rollback path: revert this commit. The parity snapshots were captured
before deletion and remain in `tests/fixtures/gradio_parity/` so a
rebuild against an older revision could verify the old behavior is
unchanged.

## Alternatives Considered

### Keep Gradio, work around its limits

- **Pros:** zero migration cost, single language (Python).
- **Cons:** every per-industry result card becomes a Gradio
  workaround; doesn't address landing-page or deep-link needs;
  visual ceiling unchanged. Doesn't fix the deploy story.
- **Rejected** because the friction was already accumulating.
  6 industries × 3-4 models each means at least 20 customised
  result cards. The cost of that scaled poorly.

### Streamlit

- **Pros:** Python-native like Gradio; a touch more layout control;
  has a reasonably active component library.
- **Cons:** Same fundamental category as Gradio: a Python "app
  framework" that visually betrays itself in every layout. Same
  deploy + deep-link problems. Routing is event-driven, not URL-
  driven; same single-page constraint.
- **Rejected** because it solves none of the listed pressures
  (visual, deep-linking, deploy) materially better than Gradio.

### Plotly Dash

- **Pros:** more layout control than Gradio; React-based under the
  hood so component library is large.
- **Cons:** still Python-server-rendered; deploy needs a Python host;
  themes look like enterprise dashboards from 2018.
- **Rejected** for the same deploy/visual reasons.

### Custom React app (Vite + React Router) instead of Next.js

- **Pros:** simpler bundler; no SSR complexity; lighter framework.
- **Cons:** loses Next.js's Server Components + ISR, which the
  `/dashboard` slice (A.9.9) depends on for the MLflow + FastAPI
  cross-fetch + 30-second cache. Loses Next.js's image and font
  optimization that show up in Core Web Vitals. Loses Vercel's
  one-click deploy story (Vercel builds Next.js apps natively).
- **Rejected** in favor of Next.js to lean on its server-side
  fetching + ISR for the dashboard.

### Next.js + Material UI (instead of shadcn/ui)

- **Pros:** mature component library; Google-style design system out
  of the box.
- **Cons:** larger bundle; more opinionated styling that's harder to
  customize per-industry; ships an entire CSS-in-JS system that
  conflicts with Tailwind.
- **Rejected** in favor of shadcn/ui: components are copy-pasted
  into the repo (no opaque dependency), Tailwind-styled (full
  customization), tree-shakable.

## Consequences

### Positive

1. **Looks like a product.** The new `/dashboard` page and per-industry
   model pages render dark-mode-first, responsive at 375 px / 768 px /
   1280 px, with a consistent shadcn visual language.
2. **Per-model deep links.** `/fintech/fraud`, `/legal/h1b-approval`,
   `/dashboard` all work as shareable URLs.
3. **Type-safe input boundary.** Every form has a Zod schema +
   typed API client (`web/lib/schemas.ts`, `web/lib/api.ts`). API
   shape drift surfaces as a TypeScript error at build time, not a
   422 in production.
4. **Public deploy story.** Static prerender (`output: "standalone"`)
   means the entire UI ships to Vercel. FastAPI deploys separately.
5. **Server Components + ISR.** The `/dashboard` page fetches FastAPI
   `/models` + MLflow run history server-side, caches for 30 s, and
   prerenders on the next access: no per-request fan-out to MLflow.
6. **`make lint` is now Python-only.** No more `app/` directory in
   the lint target list; cleaner separation of concerns.

### Negative

1. **Two languages.** Contributors now need both Python AND a TypeScript
   toolchain to run the full stack locally. Mitigated by `make
   docker-up` (one command, both stacks) and `web/README.md` (clear
   onboarding).
2. **Bigger Docker image footprint.** The new `ml-web` image adds
   ~226 MB on top of the existing `ml-api`. Mitigated by Next.js
   `output: "standalone"` (only the files the app actually uses are
   shipped) and the multi-stage Dockerfile.web (build deps not in
   the runtime image).
3. **Plotly artifacts no longer rendered in the UI.** The Gradio demand
   tab embedded a Plotly LineChart; the Next.js port uses Recharts
   instead. Visual regression accepted in SPEC §Q-A.9-1; the
   underlying forecast data is unchanged.
4. **Loss of "open and click" Gradio simplicity.** A new contributor
   can no longer launch the UI with `python app/gradio_app.py`. They
   need `make web-dev` (which spawns `pnpm dev`) or `make docker-up`.

### Neutral

- **No backend API contract change.** Both UIs called the same
  `/predict/*` and `/explain/*` endpoints. The migration didn't
  require touching `src/serving/`. The parity snapshots
  (`tests/fixtures/gradio_parity/`) prove this empirically.
- **Same models, same checkpoints.** The retirement is purely a UI
  swap. Training pipelines, MLflow registry, and on-disk
  checkpoints are untouched.

## Verification

- **Parity test:** `tests/test_parity_gradio_nextjs.py` (3 cases:
  fraud, price, demand) compares post-deletion responses against
  snapshots captured at v1.3.0-phase-a-fanout. Pass = no behavior
  drift.
- **Acceptance gates** (per SPEC §Phase A.9 success criteria):
  - `docker compose up --build` brings up exactly 3 services
    (`mlflow`, `ml-api`, `ml-web`): all healthy ≤120 s. ✓
  - `git grep -i gradio` returns matches only in this ADR + the
    superseded SPEC sections (kept for archaeology). Zero matches
    in `src/`, `web/`, `Makefile`, `docker-compose*.yml`. ✓
  - `make test` still 360+ tests passing, ≥90 % coverage. ✓
  - `make lint` and `pnpm lint` both 0 warnings. ✓

## References

- `SPEC.md` §Phase A.9: full retirement spec
- `tasks/plan.md` §Phase A.9: task-level breakdown
- `tests/test_parity_gradio_nextjs.py`: the regression gate
- `web/README.md`: Next.js app onboarding
- Tag `v1.3.0-phase-a-fanout`: last commit with Gradio in tree
- Tag `v1.4.0-phase-a-complete`: first commit without Gradio (TBD A.9.12)
