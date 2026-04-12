# Design Decisions — Portfolio ML System Production Hardening

> Each decision documents what was chosen, what alternatives were considered, and why. Decisions were made independently based on codebase analysis, portfolio goals, and production ML best practices.

---

## Decision 1: Experiment Tracking — Keep W&B + Add MLflow

**Date:** 2026-04-11

**Context:** Training code uses W&B with graceful fallback. docker-compose.yml has an MLflow service that no code connects to. Both are industry-standard tools.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) Replace W&B with MLflow | Single tool, self-hosted | Loses W&B integration (already working), less portfolio breadth |
| B) Keep W&B only | No new code needed | MLflow service in compose is dead weight, no model registry |
| C) Both: MLflow primary + W&B optional | Shows breadth, MLflow self-hosted (no API key), model registry | More code to maintain, two tracking UIs |

**Chosen: C — Dual tracking with MLflow as primary**

MLflow is always available (self-hosted, zero config). W&B remains optional for users with API keys. This is the most common pattern in enterprise ML teams — MLflow for model registry/lifecycle, W&B for rich experiment visualization. Having both in a portfolio demonstrates familiarity with the two dominant tools. MLflow's model registry (versioning, staging, production promotion) is a capability W&B free tier doesn't offer.

---

## Decision 2: Docker Strategy — Separate Multi-Stage Dockerfiles

**Date:** 2026-04-11

**Context:** docker-compose.yml references `Dockerfile.api` and `Dockerfile.ui` that don't exist. Need to create them.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) Single Dockerfile with build args | DRY, one file to maintain | Complex build args, cache invalidation issues |
| B) Separate Dockerfiles (api + ui) | Matches compose expectations, independent optimization | Some duplication in dependency install |
| C) Base Dockerfile extended by api/ui | Maximum DRY | Three files for two services, Docker multi-FROM complexity |

**Chosen: B — Separate Dockerfiles**

docker-compose.yml already references `Dockerfile.api` and `Dockerfile.ui` — matching existing expectations is the simplest path. Each uses multi-stage (builder → runtime) to keep images small. The dependency install pattern is identical but the duplication is minimal (~5 lines) and each can be independently optimized (e.g., UI might not need all ML deps). This is the most readable and maintainable approach.

---

## Decision 3: PyTorch in Docker — CPU Only

**Date:** 2026-04-11

**Context:** Host Mac uses MPS (Apple Silicon GPU). Docker containers run Linux. Need to decide GPU strategy for containers.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) CPU-only PyTorch | Works everywhere, smaller images | Slower training in Docker |
| B) NVIDIA GPU passthrough | GPU-accelerated | Requires Linux + NVIDIA hardware (not Mac) |
| C) MPS in Docker | Native GPU on Mac | Impossible — MPS is macOS-only, not in Linux containers |

**Chosen: A — CPU-only**

MPS is impossible in Docker (macOS-only). NVIDIA GPU requires Linux host with NVIDIA hardware. CPU is the only viable option for Docker on Mac. The existing `src/device.py` already handles CPU fallback gracefully — `get_device()` returns CPU when MPS is unavailable, and `map_location=self.device` in the predictor handles checkpoint loading. CPU inference is <10ms for all models. Training in Docker is slower but acceptable for a portfolio demo. Host-native training with MPS remains available via `make train`.

---

## Decision 4: Docker-Compose — Self-Contained Dev Stack

**Date:** 2026-04-11

**Context:** Current docker-compose.yml references PostgreSQL, Redis, and an external `backend` network that don't exist standalone. It's impossible to run `docker compose up` without those external services.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) Keep PostgreSQL dependency | Production-like | Requires running PostgreSQL separately, heavy |
| B) SQLite for dev, PostgreSQL for prod | Self-contained dev, prod-ready upgrade path | Two compose files |
| C) Remove MLflow entirely | Simplest | Loses experiment tracking, wastes existing setup |

**Chosen: B — SQLite for dev, PostgreSQL for prod**

docker-compose.dev.yml already configures SQLite for MLflow. The fix: make the dev setup the default `docker-compose.yml` (self-contained, SQLite, no external deps), and preserve the current production config as `docker-compose.prod.yml` (PostgreSQL, Traefik labels, external network). This follows Docker best practices: `docker compose up` should work out of the box for any developer cloning the repo.

---

## Decision 5: Test Fixtures — Tiny Real Models

**Date:** 2026-04-11

**Context:** Tests need model checkpoints. Current tests use real 5.4MB checkpoints from `checkpoints/` directory, which requires prior training and are not in git.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) Real trained checkpoints | Tests exercise real model behavior | 5.4MB, brittle (require training), slow to load |
| B) Tiny real models in conftest.py | Fast, deterministic, test real code paths | Need to maintain fixture factories |
| C) Mock all model interfaces | Fastest, no model dependencies | Hides real bugs (shape mismatches, device errors) |

**Chosen: B — Tiny real models in conftest.py**

Previous experience (project 05) showed that mocked tests pass while production fails — the mock/prod divergence problem. Tiny models (50 rows of data, 2 epochs) are fast (<1s creation), deterministic (seeded), and exercise the real code paths including tensor operations, feature engineering, and checkpoint loading. They catch shape mismatches and device errors that mocks would miss. Saved to `tmp_path` for automatic cleanup.

---

## Decision 6: Model Explainability — SHAP for Trees, Gradients for Neural Nets

**Date:** 2026-04-11

**Context:** All 4 models produce predictions with no explanation. Adding explainability is a high-impact portfolio differentiator.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) SHAP only | Gold standard for tree models | KernelExplainer for neural nets is extremely slow |
| B) SHAP for trees + gradient-based for neural nets | Fast, model-appropriate techniques | Different API per model type |
| C) LIME universal | Single API for all models | Slower, less precise than SHAP for trees, adds dependency |

**Chosen: B — SHAP for trees, gradient-based for neural nets**

Different model families need different explainability techniques — showing this understanding is itself a portfolio signal. SHAP TreeExplainer is the gold standard for XGBoost/LightGBM: fast, exact, well-understood. For the fraud autoencoder, gradient-based feature importance (which features contribute most to reconstruction error) is more natural and faster than SHAP KernelExplainer. Demand forecasting LSTM is excluded — temporal attention would require model architecture changes (out of scope).

**Coverage:**
- Credit Risk (XGBoost): SHAP TreeExplainer → waterfall + bar chart
- Price Prediction (LightGBM): SHAP TreeExplainer → waterfall + bar chart
- Fraud Detection (Autoencoder): Gradient-based attribution → feature ranking
- Demand Forecasting (LSTM): Not included (would require attention layers)

---

## Decision 7: Structured Logging — stdlib with JSON Formatter

**Date:** 2026-04-11

**Context:** All output is via `rich.console.print()` — great for terminal but useless for Docker log aggregation.

**Options:**
| Option | Pros | Cons |
|--------|------|------|
| A) structlog | Rich structured logging, processors, context binding | New dependency |
| B) stdlib logging + JSON formatter | Zero dependencies, universal integration | More manual setup |
| C) loguru | Simple API, pretty output | Opinionated, doesn't integrate well with uvicorn/MLflow |

**Chosen: B — stdlib logging with JSON formatter**

Zero new dependencies. stdlib logging is universally understood and integrates with every Python library (FastAPI, uvicorn, MLflow all use it). A simple JSON formatter class (~20 lines) is all that's needed. structlog is excellent but adds a dependency for marginal benefit. In Docker, JSON logs are the standard for log aggregation (ELK, CloudWatch, Datadog). Rich console output is preserved for Gradio UI only (user-facing, not log output).

**Format switching:** `LOG_FORMAT=json` (Docker default) or `LOG_FORMAT=text` (host default). Controlled by environment variable.
