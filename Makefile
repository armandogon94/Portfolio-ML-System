.PHONY: setup data train evaluate ui serve test lint clean all

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
