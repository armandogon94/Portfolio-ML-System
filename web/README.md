# Portfolio ML System — Frontend

Next.js 14 + shadcn/ui + TanStack Query + Tailwind v3. The web UI for the industry-specific ML demo system. Talks to FastAPI (`../src/serving/api.py`) via Next.js rewrites — no CORS, no backend middleware.

> See the parent [README](../README.md), [SPEC.md §"Phase A.2"](../SPEC.md), and [`tasks/plan.md`](../tasks/plan.md) for the broader context.

## Prerequisites

- **Node 20 LTS** (`node --version` → `v20.x`)
- **pnpm 10** — `npm install -g pnpm` or `brew install pnpm`
- **Docker + Docker Compose** (only if you want the containerized workflow)

For the full-stack backend you also need the root project set up — see the root [README](../README.md#quick-start).

## Three ways to run

Pick the one that matches what you're doing.

### 1. Native dev (fastest — recommended for day-to-day work)

Starts Next.js with hot-reload on http://localhost:3070 and proxies `/api/*` to FastAPI on `localhost:8070`.

```bash
# From repo root (requires the backend running separately):
make serve         # terminal 1: FastAPI at :8070
make web-dev       # terminal 2: Next.js at :3070

# Or from web/:
INTERNAL_API_URL=http://localhost:8070 pnpm dev
```

Open http://localhost:3070 and you'll see the landing page. Click a tile → industry index → model page.

### 2. Production Docker (simulates deploy)

Builds every service from scratch, matches the production image exactly, slower iteration.

```bash
# From repo root:
make docker-up     # brings up the 3-service stack: mlflow, ml-api, ml-web
```

Visit http://localhost:3070 (industry pages + `/dashboard`).

### 3. Dockerized dev (offline HMR fallback)

`ml-web` runs `pnpm dev` inside the container with `./web` volume-mounted. Same HMR experience as native dev, but works without Node installed locally.

```bash
make docker-dev-up
```

Slower than native due to container filesystem overhead — only use when native dev isn't an option.

## Common commands

From the repo root:

```bash
make web-install   # pnpm install
make web-dev       # pnpm dev (port 3070)
make web-build     # pnpm build (production)
make web-test      # pnpm test (Vitest)
make web-lint      # pnpm lint + pnpm typecheck
make web-typecheck # pnpm typecheck only
```

From `web/`:

```bash
pnpm dev           # dev server at :3070
pnpm build         # production build (output: "standalone")
pnpm start         # serve the production build at :3070
pnpm test          # Vitest once
pnpm test:watch    # Vitest watch mode
pnpm test:coverage # Vitest with v8 coverage report
pnpm lint          # next lint
pnpm typecheck     # tsc --noEmit
pnpm format        # prettier --write .
```

## Project layout

```
web/
├── app/                           # Next.js App Router
│   ├── page.tsx                   # landing page (6 industry tiles)
│   ├── layout.tsx                 # Nav + Providers + fonts
│   ├── providers.tsx              # QueryClient + Theme + Toaster
│   ├── globals.css                # Tailwind + shadcn CSS variables
│   └── <industry>/                # 6 industry routes
│       ├── page.tsx               # index (stub until A.3-A.8)
│       └── <model>/page.tsx       # live model demo (A.2.7+)
├── components/
│   ├── ui/                        # shadcn primitives (owned source)
│   ├── ModelForm.tsx              # Reusable form over Zod schema
│   ├── PredictionResult.tsx       # Score / rec / confidence card
│   ├── ExplainabilityChart.tsx    # SHAP Recharts horizontal bar
│   ├── IndustryTile.tsx           # Landing card
│   ├── IndustryIndex.tsx          # Shared layout for /<industry>
│   ├── Nav.tsx                    # Top bar
│   └── ThemeToggle.tsx            # Dark-mode switch
├── lib/
│   ├── api.ts                     # Typed FastAPI client + Zod validation
│   ├── schemas.ts                 # Zod input schemas per model
│   ├── industries.ts              # 6-industry registry
│   ├── query-client.ts            # Singleton TanStack QueryClient
│   └── utils.ts                   # cn() helper (clsx + tailwind-merge)
└── __tests__/                     # Vitest suites (by module)
```

## Environment variables

Copy `.env.example` → `.env.local` and set what you need. The proxy makes both optional in most setups.

- `INTERNAL_API_URL` — server-side rewrite destination for `/api/*`.
  - Local dev: `http://localhost:8070` (FastAPI on host)
  - Docker:    `http://ml-api:8000` (FastAPI inside compose network)
- `NEXT_PUBLIC_API_URL` — reserved for future direct client-side calls; currently unused.

## Adding a new model (Phases A.3–A.8 pattern)

After A.2 ships, each new industry model follows the same pattern. Run through this checklist:

1. **Schema** — append a Zod schema + `<Model>_DEFAULTS` to `web/lib/schemas.ts` under your industry section.
2. **API function** — append `predict<Model>` (and optionally `explain<Model>`) to `web/lib/api.ts` using the `post()` helper with response Zod schemas.
3. **Fields** — create `web/app/<industry>/<model>/fields.ts` exporting a `FieldConfig<Input>[]`.
4. **Page** — create `web/app/<industry>/<model>/page.tsx` by copying `web/app/fintech/credit-risk/page.tsx` and swapping imports.
5. **Registry** — flip `ready: true` on the corresponding entry in `web/lib/industries.ts` (or add a new one).
6. **Tests** — Vitest smoke test under `__tests__/app/<model>.test.tsx` mocking lib/api.

The shared `IndustryIndex` component picks up new `ready` models automatically — no edits to industry-index pages required.

## Troubleshooting

**`pnpm lint` fails with "Unknown options: useEslintrc, extensions..."**
ESLint got bumped to v10 (incompatible with Next 14's `next lint`). Pin it: `pnpm remove eslint && pnpm add -D eslint@^8`.

**`ResizeObserver is not defined` in Vitest**
Already polyfilled in `vitest.setup.ts` for shadcn Slider/Switch. If a new Radix component needs more, add the polyfill there.

**Dark mode flashes light on load**
Expected on first page load before next-themes applies the class. `disableTransitionOnChange` is set; a more aggressive solution is the flicker-free script — revisit if it bothers the demo experience.

**Proxy fails with 502/504**
FastAPI isn't running. Start it with `make serve` (native) or ensure `ml-api` is healthy in Docker (`docker compose ps`).
