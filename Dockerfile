# QuantX Engine REST API server
# Build: docker build -t quantx-server .
# Run:   docker run -p 8080:8080 -v /path/to/quantx.dll:/app/lib/quantx.dll -e QUANTX_LIB_PATH=/app/lib/quantx.dll quantx-server

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/pyproject.toml ./server/pyproject.toml
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

# Copy Python SDK
COPY python/ ./python/

# Copy server code
COPY server/ ./server/

EXPOSE 8080

ENV QUANTX_LIB_PATH=/app/lib/quantx.so

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
