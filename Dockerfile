# Dockerfile - Optimized multi-stage build for Flask API
# Build time: ~4-5 min (cold), ~2 min (cached)
# Image size: ~650-700MB

# ============================================
# Stage 1: Builder (compilation environment)
# ============================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies (needed for compiling spacy, onnxruntime, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements for better layer caching
COPY requirements.txt .

# Create virtual env and install dependencies with pip cache mount
# Cache mount persists between builds for faster re-builds
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime (production environment)
# ============================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install ONLY runtime dependencies (NOT build-essential)
# libgomp1: Required by onnxruntime and some numpy operations
# curl: For health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder (includes all Python packages)
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy application code
COPY --chown=appuser:appgroup . .

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

USER appuser

CMD ["python", "app.py"]
