.PHONY: setup data train evaluate ui serve test lint format clean all
.PHONY: docker-build docker-up docker-down docker-logs docker-test docker-clean docker-dev-up docker-dev-down
.PHONY: web-install web-dev web-build web-test web-lint web-typecheck

setup:
	uv sync --extra dev

data:
	uv run python scripts/generate_data.py --problem all

train:
	uv run python scripts/train.py --model all

evaluate:
	uv run python scripts/evaluate.py --model all

ui:
	uv run python app/gradio_app.py

serve:
	uv run python scripts/serve.py

test:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/ scripts/ app/ tests/
	uv run ruff format --check src/ scripts/ app/ tests/

format:
	uv run ruff format src/ scripts/ app/ tests/

clean:
	rm -rf data/raw/*.csv data/processed/*.csv
	rm -rf checkpoints/*/
	rm -rf results/*.csv
	rm -rf wandb/
	rm -rf __pycache__ .pytest_cache

all: data train evaluate
	@echo "Full pipeline complete. Run 'make ui' to launch the web interface."

# ── Docker targets ──────────────────────────────────────────────────

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

# Dev override — ml-web runs `pnpm dev` with volume-mounted source
# so file edits hot-reload in the container. Other services
# (mlflow, ml-api, ml-ui) stay production-style.
docker-dev-up:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

docker-dev-down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

# ── Next.js frontend targets (web/) ─────────────────────────────────
# Prefer these for day-to-day dev — native pnpm is faster than Docker.
# INTERNAL_API_URL is the server-side rewrite target (see web/next.config.mjs).

web-install:
	cd web && pnpm install

web-dev:
	cd web && INTERNAL_API_URL=http://localhost:8070 pnpm dev

web-build:
	cd web && pnpm build

web-test:
	cd web && pnpm test

web-lint:
	cd web && pnpm lint && pnpm typecheck

web-typecheck:
	cd web && pnpm typecheck
