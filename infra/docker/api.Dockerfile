# =============================================================================
# api.Dockerfile — FastAPI inference server.
# Multi-stage: builder installs deps, runtime carries only what serving needs.
#
# Build context is the REPOSITORY ROOT:
#   docker build -f infra/docker/api.Dockerfile .
#
# This build fails immediately if uv.lock is not committed, because of the
# COPY + `uv sync --frozen` pair below. That is deliberate: it is the check that
# caught the lockfile being gitignored while the README advertised the build.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies with uv
# ---------------------------------------------------------------------------
FROM python:3.11.15-slim AS builder

# Install system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself — source copied in runtime)
RUN uv sync --frozen --no-dev --no-install-project

# CPU-only PyTorch. MPS is macOS-only and cannot exist in a Linux container;
# CUDA would need an NVIDIA host. The README states this tradeoff explicitly
# rather than implying the container is GPU-accelerated.
RUN uv pip install "torch==2.11.0" \
    --index-url https://download.pytorch.org/whl/cpu \
    --reinstall \
    --no-deps

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal image with only what's needed
# ---------------------------------------------------------------------------
FROM python:3.11.15-slim AS runtime

# Install runtime system dependencies (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser -d /app -s /sbin/nologin mluser

WORKDIR /app

# Copy virtual environment from builder (with ownership set in COPY to avoid extra layer)
COPY --from=builder --chown=mluser:mluser /app/.venv /app/.venv

# Ensure venv binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# Application code.
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser configs/ ./configs/
COPY --chown=mluser:mluser scripts/serve.py ./scripts/serve.py

# The committed CI fixtures. Small, synthetic, and never used for a published
# number — see data/README.md. They let the container's e2e path run offline.
COPY --chown=mluser:mluser data/sample/ ./data/sample/

# Checkpoints are NOT copied. They are gitignored, so on a fresh clone the
# directory does not exist and a COPY here would fail the build. They arrive at
# runtime through the read-only bind mount in infra/compose/base.yml. Created
# empty so the registry's glob has a directory to find.
RUN mkdir -p /app/checkpoints && chown mluser:mluser /app/checkpoints

USER mluser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
