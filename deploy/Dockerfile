# Multi-stage Dockerfile for the QuantX API server
# Build:  docker build -f deploy/Dockerfile -t quantx-api .
# Run:    docker run -p 8000:8000 quantx-api

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first (layer caching)
COPY python/pyproject.toml python/
COPY server/pyproject.toml server/

# Install Python deps into /install prefix
RUN pip install --upgrade pip && \
    pip install --prefix=/install \
        scipy networkx pennylane \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "aiosqlite>=0.19.0" \
        "redis>=5.0" \
        "prometheus_client>=0.19.0"

# Copy source
COPY python/ python/
COPY server/ server/

# Install the SDK and server into /install
RUN pip install --prefix=/install --no-deps -e python/ && \
    pip install --prefix=/install --no-deps -e server/

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN addgroup --system quantx && adduser --system --ingroup quantx quantx

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source (editable install already wired up)
COPY --from=builder /build/python /app/python
COPY --from=builder /build/server /app/server

# Ensure server and SDK are on sys.path
ENV PYTHONPATH=/app

USER quantx

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
