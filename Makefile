# Fintech ML System — one entrypoint per stage.
#
# The default target is `help`, so `make` on its own tells you what exists
# instead of silently running the first rule.

COMPOSE := docker compose -f infra/compose/base.yml
COMPOSE_DEV := $(COMPOSE) -f infra/compose/dev.yml
MMDC ?= npx -y @mermaid-js/mermaid-cli@11.16.0

.DEFAULT_GOAL := help
.PHONY: help setup data train train-sample evaluate figures diagrams diagrams-check \
        screenshots-install screenshots serve \
        test test-all lint format typecheck verify clean \
        docker-build docker-up docker-down docker-logs docker-clean \
        docker-dev-up docker-dev-down \
        web-install web-dev web-build web-test web-lint web-typecheck

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Python pipeline ─────────────────────────────────────────────────────────

setup:  ## Install Python deps from the committed lockfile
	uv sync --frozen --extra dev

data:  ## Download all real datasets (3 need Kaggle; ULB/OpenML does not)
	uv run python scripts/download_data.py --dataset all

train:  ## Train every dataset config on real data (4 configs, 3 problems)
	uv run python scripts/train.py --model all

train-sample:  ## Smoke-train on the committed CI fixtures (writes NO checkpoint)
	uv run python scripts/train.py --model all --sample

evaluate:  ## Print the results table, read from reports/*_metrics.csv
	uv run python scripts/evaluate.py

figures:  ## Regenerate the published PR and calibration figures from OOF scores
	uv run python scripts/make_figures.py --published-only

diagrams:  ## Export docs/diagrams/*.mmd to SVG
	@set -e; for f in docs/diagrams/*.mmd; do \
	  echo "  $$f"; \
	  $(MMDC) -i "$$f" -o "$${f%.mmd}.svg" -b transparent; \
	done

diagrams-check:  ## Render every Mermaid source and verify text containment
	MMDC_BIN="$(MMDC)" uv run python scripts/check_diagram_text.py

screenshots:  ## Capture README screenshots (needs the stack running + a trained model)
	uv run --extra dev python scripts/capture_screenshots.py

screenshots-install:  ## Install the Chromium browser used by the screenshot script
	uv run --extra dev playwright install chromium

serve:  ## Run the inference API on :8070
	uv run python scripts/serve.py

# ── Quality gates ───────────────────────────────────────────────────────────

test:  ## Run the test suite (excludes the live-Kaggle canary)
	uv run pytest -m "not network"

test-all:  ## Run everything INCLUDING the live-Kaggle canary (needs credentials)
	uv run pytest

lint:  ## ruff check + format check
	uv run ruff check src/ scripts/ tests/
	uv run ruff format --check src/ scripts/ tests/

format:  ## Apply ruff formatting
	uv run ruff format src/ scripts/ tests/

typecheck:  ## mypy on src/
	uv run mypy src/

verify:  ## Clone HEAD into a temp dir and run the documented quickstart
	./scripts/verify_fresh_clone.sh

clean:  ## Remove generated data, checkpoints and caches
	rm -rf data/raw/*.csv data/processed/* checkpoints/*/ reports/figures/*.png
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov __pycache__

# ── Docker ──────────────────────────────────────────────────────────────────

docker-build:  ## Build all images
	$(COMPOSE) build

docker-up:  ## Start the full stack (mlflow :5070, api :8070, web :3070)
	$(COMPOSE) up -d --wait
	@echo "  web      http://localhost:3070"
	@echo "  api      http://localhost:8070/docs"
	@echo "  mlflow   http://localhost:5070"

docker-down:  ## Stop the stack
	$(COMPOSE) down

docker-logs:  ## Tail the stack logs
	$(COMPOSE) logs -f

docker-clean:  ## Stop the stack and remove its volumes and local images
	$(COMPOSE) down -v --rmi local

docker-dev-up:  ## Stack with the web service in hot-reload mode
	$(COMPOSE_DEV) up

docker-dev-down:
	$(COMPOSE_DEV) down

# ── Next.js frontend ────────────────────────────────────────────────────────
# Prefer these over Docker for day-to-day work; native pnpm is much faster.

web-install:
	cd web && pnpm install --frozen-lockfile

web-dev:  ## Next.js dev server on :3070, proxying /api to :8070
	cd web && INTERNAL_API_URL=http://localhost:8070 pnpm dev

web-build:
	cd web && pnpm build

web-test:
	cd web && pnpm test

web-lint:
	cd web && pnpm lint && pnpm typecheck

web-typecheck:
	cd web && pnpm typecheck
