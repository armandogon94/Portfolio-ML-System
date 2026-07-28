# Port convention

**Placeholder values only.** Real deployment topology, if this project ever has
any, belongs in `ops/ports.local.md`, which is gitignored. Publishing a
cross-project port map tells a reader more about the author's machine than about
the project.

## The scheme

For a project numbered `NN`:

| Service | Pattern | This project (`NN = 07`) |
|---|---|---|
| Frontend / web UI | `3NN0`–`3NN9` | `3070` |
| Backend API | `8NN0`–`8NN9` | `8070` |
| PostgreSQL | `54NN` | `5407` *(unused: this project has no database)* |
| Redis | `63NN` | `6307` *(unused)* |
| Extra services | see below | `5070` (MLflow) |

The repo-local defaults above are real and copy-pasteable, because the
Quickstart in `README.md` has to work as written. What is deliberately *not*
published is the allocation for every other project on the author's machine.

## Never bind these

| Port | Why |
|---|---|
| `5000`, `7000` | macOS **AirPlay Receiver** binds both. A container appears to start and is then unreachable: a debugging session spent on a non-bug. |
| `11434` | Ollama. |
| `5432`, `6379` | Postgres and Redis defaults. They collide with any local install and with every other project that lazily took the default. |
| `3000`, `8000` | Next.js and FastAPI/uvicorn defaults. Same reason. |
| `80`, `443` | Reserved for a local reverse proxy. |

`scripts/serve.py` refuses to start on any of these and says why.

## Overriding

Every port is an environment variable with a default, so nothing is hardcoded:

```bash
# .env  (copy from .env.example)
WEB_PORT=<3NN0>
BACKEND_PORT=<8NN0>
MLFLOW_TRACKING_PORT=<5NN0>
```

```yaml
# infra/compose/base.yml
ports:
  - "${BACKEND_PORT:-8070}:8000"
```

Note the asymmetry: **inside** a container the service binds the framework
default (`8000`), because container network namespaces are isolated and
collisions are impossible there. Only the host-side port needs allocating.

## Checking before you start

```bash
lsof -nP -iTCP:8070 -sTCP:LISTEN
lsof -nP -iTCP:3070 -sTCP:LISTEN
lsof -nP -iTCP:5070 -sTCP:LISTEN
```

Each should print nothing, or only this project's own processes.
