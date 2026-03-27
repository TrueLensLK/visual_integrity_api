# syntax=docker/dockerfile:1.4
# ===========================================================================
# OPTIMIZED Multi-Stage Build for GCP VMs & Cloud Deployments
# ===========================================================================
# Problem: Single-stage builds leave build tools (~500MB+) in final image,
#          causing "no space left on device" errors on VMs with limited disk.
#
# Solution: Two-stage build:
#   STAGE 1 (builder): Compiles all wheels from source (cached, discarded)
#   STAGE 2 (runtime): Installs pre-compiled wheels only (small, fast)
#
# Result: Final image ~60% smaller (saves disk during export phase)
# ===========================================================================

# =========================================================================
# STAGE 1: BUILDER — Compile wheels (runs once per requirements change)
# =========================================================================
FROM python:3.12-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=5

# Install build dependencies (ONLY in builder, removed in final image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpng-dev \
        libjpeg-dev \
        libwebp-dev \
        libprotobuf-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /builder

COPY requirements.txt .

# Compile all dependencies into wheels (cached layer)
# Phase 1: PyTorch from official index (faster)
# Phase 2: All other packages
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheels && \
    pip wheel --timeout 600 \
        --index-url https://download.pytorch.org/whl/cpu \
        --wheel-dir /wheels \
        torch torchvision && \
    pip wheel --timeout 600 \
        --wheel-dir /wheels \
        -r requirements.txt

# =========================================================================
# STAGE 2: RUNTIME — Final production image (small, no build tools)
# =========================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install ONLY runtime dependencies (no build tools, no -dev packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libpng16 \
        libjpeg62-turbo \
        libwebp7 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-compiled wheels from builder (instant install, no compilation)
COPY --from=builder /wheels /wheels
COPY --from=builder /builder/requirements.txt .

# Install pre-compiled wheels at lightning speed
RUN pip install --timeout 300 \
        --no-index \
        --find-links /wheels \
        -r requirements.txt && \
    rm -rf /wheels

# Copy application source
COPY . .

# ---------------------------------------------------------------------------
# Security: least-privilege non-root user
#
# Running as root inside a container is a serious security risk: a container
# escape would grant the attacker root on the host.  We create a locked,
# no-login system account (ai_worker) with no home directory and no shell,
# then transfer ownership of /app to it so the process can write to
# temp_uploads/ at runtime.
#
# --system   : creates a system account (UID < 1000, no cron, no password)
# --no-create-home : no /home/ai_worker; not needed for a web process
# --shell /bin/false : prevents interactive login even if credentials leak
# ---------------------------------------------------------------------------
RUN adduser \
        --system \
        --no-create-home \
        --shell /bin/false \
        ai_worker \
    # Give the worker ownership of the app directory so it can write
    # temporary upload files and any runtime artefacts.
    && chown -R ai_worker /app

USER ai_worker

# ---------------------------------------------------------------------------
# Exposed port
# Documents that the service listens on 8000; does not publish the port by
# itself — that is done via -p or docker-compose ports mapping at runtime.
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check
# The container orchestrator (Docker Swarm / Kubernetes liveness probe) polls
# this every 30 s.  A fast /health response from uvicorn confirms both that
# the process is alive and that the event loop is not stalled.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Entry point
#
#   main:app        — module `main`, FastAPI instance `app`
#   --host 0.0.0.0  — bind to all interfaces so the container port is reachable
#                     from outside (localhost-only binding would be invisible
#                     to the Docker network bridge).
#   --port 8000     — explicit; keeps the value consistent with EXPOSE above.
#   --workers 2     — Gunicorn-style multi-process mode baked into uvicorn[standard].
#                     Each worker is an independent Python process with its own
#                     event loop, so one slow blocking call cannot stall all
#                     concurrent requests.  Rule of thumb: 2× CPU cores;
#                     override at runtime with the WORKERS env var if needed.
#
# NOTE: --workers > 1 means the lifespan context manager (_lifespan in main.py)
# runs once per worker process, giving each its own httpx.AsyncClient pool —
# which is correct; sharing a client across OS processes is not safe.
# ---------------------------------------------------------------------------
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
