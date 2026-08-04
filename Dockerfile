# syntax=docker/dockerfile:1
#
# Multi-stage build.
#
# Stage 1 compiles wheels with a full toolchain. Stage 2 copies only the wheels
# into a slim base, so gcc/g++ and the build caches never reach the shipped
# image. That is the difference between roughly 1.1 GB and roughly 400 MB, and
# it removes a compiler from the runtime attack surface.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libgomp1 is the OpenMP runtime LightGBM links against. Without it the image
# builds cleanly and then dies at `import lightgbm` with a missing .so — the
# single most common failure when moving LightGBM onto a slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    ARTIFACT_DIR=/data/artifacts

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Run as a non-root user. The numeric UID is set explicitly so the Kubernetes
# securityContext can reference it without depending on image internals.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/artifacts \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

# Probes the readiness endpoint, not the process. A container that is up but
# has no artifacts loaded is not serving anything useful.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "ran_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
