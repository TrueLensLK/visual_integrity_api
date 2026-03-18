# syntax=docker/dockerfile:1.4
# ===========================================================================
# Stage: runtime
# Base: python:3.12-slim — matches the venv Python version in use (3.12.7),
#       slim variant strips test suites, manpages and locale data that the
#       full image ships, cutting ~200 MB from the base layer alone.
# ===========================================================================
FROM python:3.12-slim

# ---------------------------------------------------------------------------
# Environment hardening
#   PYTHONDONTWRITEBYTECODE=1  – suppresses .pyc file generation; irrelevant
#                                at runtime and wastes inode quota in the
#                                container overlay filesystem.
#   PYTHONUNBUFFERED=1         – forces stdout/stderr to flush immediately so
#                                uvicorn logs reach the container runtime
#                                (Docker, K8s) without buffering delay.
#   PIP_DISABLE_PIP_VERSION_CHECK=1 – suppresses the "new pip available"
#                                     noise that pollutes build logs.
#   PIP_DEFAULT_TIMEOUT=300    – increase socket timeout to 5 minutes to
#                                handle slow PyPI downloads (default is 15s).
#   PIP_RETRIES=5              – retry failed downloads up to 5 times to
#                                handle transient network issues.
#
# NOTE: We intentionally DO NOT set PIP_NO_CACHE_DIR because we use BuildKit
# cache mounts (--mount=type=cache) to persist downloaded wheels across builds.
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# ---------------------------------------------------------------------------
# System dependencies — all installed and cache-purged in ONE RUN layer.
#
# Each extra RUN would add a new overlay layer; combining them means Docker
# never commits an intermediate layer that still contains the apt cache.
#
#   build-essential / gcc   — compile C extensions: pdqhash, numpy, Pillow,
#                             scipy, and OpenCV's Python bindings all need a
#                             C compiler at wheel-build time.
#   libgl1                  — OpenGL runtime required by OpenCV (cv2).
#   libglib2.0-0            — GLib required by OpenCV and mediapipe.
#   libgomp1                — OpenMP used by numpy/scipy for parallel BLAS.
#   libsm6 / libxext6 /
#   libxrender1             — X11 stubs needed by some OpenCV codepaths even
#                             in headless mode.
#   libpng-dev / libjpeg-dev
#   / libwebp-dev           — Pillow compiles against these to enable native
#                             PNG / JPEG / WebP codec support.
#   libprotobuf-dev         — protobuf C++ runtime for mediapipe & TFLite.
#   curl                    — lightweight health-check probe (used by the
#                             Docker HEALTHCHECK below).
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
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
        curl \
    # Purge apt lists in the same layer — if split into a separate RUN the
    # lists are already committed to the layer above and the purge saves
    # nothing in the final image.
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# /app is the conventional path for containerised web services; using an
# absolute path avoids any ambiguity about relative operations later.
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Layer-cache optimisation — dependency installation
#
# COPY requirements.txt first, then pip install, THEN copy source code.
# Docker's layer cache is keyed on file content: as long as requirements.txt
# is unchanged, the `pip install` layer is reused on every rebuild, even when
# application source files change.  This is the single most impactful caching
# technique for Python images — it saves minutes per CI build.
#
# NOTE: Large packages (torch, transformers) can timeout on slow networks.
# The PIP_DEFAULT_TIMEOUT=300 and PIP_RETRIES=5 env vars handle this, but
# we also add explicit --timeout flag for extra safety.
# ---------------------------------------------------------------------------
COPY requirements.txt .

# Install dependencies using BuildKit cache mounts.
# --mount=type=cache persists the pip download cache across builds, so even if
# the layer is invalidated, previously downloaded wheels don't need re-downloading.
# This is critical when building from Git URL contexts where layer caching is weak.
#
# Phase 1: Install PyTorch separately (largest packages ~1GB+, most likely to timeout)
# Phase 2: Install remaining dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 600 \
        "torch>=2.1.0" \
        "torchvision>=0.16.0" \
    && pip install --timeout 300 -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# Copied AFTER the dependency layer so that code changes don't invalidate the
# expensive pip install cache.
# ---------------------------------------------------------------------------
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
