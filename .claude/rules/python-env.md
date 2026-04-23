# Python Environment Rule

ALWAYS use `uv` for Python virtual environments and dependency management. NEVER use conda, miniconda, pip directly, virtualenv, or python -m venv.

- Install deps: `uv sync` or `uv sync --extra dev`
- Run scripts: `uv run python script.py`
- Run tests: `uv run pytest`
- Add dependency: `uv add package-name`
- Create new project: `uv init`

If a project has a `requirements.txt` but no `pyproject.toml`, migrate it:
```bash
uv init
uv add $(cat requirements.txt)
```
