"""M107: Prometheus metrics — counters, histograms, and gauges for QuantX.

All metric objects are created only when ``prometheus_client`` is installed.
When the package is absent every public helper is a no-op so the server
runs correctly without the optional dependency.

Install::

    pip install prometheus_client
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

# ---------------------------------------------------------------------------
# Optional import
# ---------------------------------------------------------------------------

try:
    import prometheus_client as _prom
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CONTENT_TYPE_LATEST,
        generate_latest,
    )
    PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric definitions (created only when prometheus_client is available)
# ---------------------------------------------------------------------------

if PROM_AVAILABLE:
    REQUEST_COUNTER: Any = Counter(
        "quantx_requests_total",
        "Total HTTP requests handled by the QuantX API",
        ["endpoint", "status"],
    )
    REQUEST_DURATION: Any = Histogram(
        "quantx_request_duration_seconds",
        "HTTP request processing time in seconds",
        ["endpoint"],
    )
    JOB_COUNTER: Any = Counter(
        "quantx_jobs_total",
        "Total async jobs by terminal status",
        ["status"],
    )
    JOB_DURATION: Any = Histogram(
        "quantx_job_duration_seconds",
        "Async job processing time from submit to completion, by endpoint",
        ["endpoint"],
    )
    CACHE_SIZE: Any = Gauge(
        "quantx_cache_size",
        "Current number of entries in the result cache",
    )
    JOBS_ACTIVE: Any = Gauge(
        "quantx_jobs_active",
        "Number of jobs currently in the 'running' state",
    )
else:  # pragma: no cover
    # Provide dummy objects so type-checkers are happy
    REQUEST_COUNTER = None
    REQUEST_DURATION = None
    JOB_COUNTER = None
    JOB_DURATION = None
    CACHE_SIZE = None
    JOBS_ACTIVE = None


# ---------------------------------------------------------------------------
# Public helpers — safe no-ops when prometheus_client is absent
# ---------------------------------------------------------------------------

def track_job_status(status: str) -> None:
    """Increment ``quantx_jobs_total`` for the given *status* label."""
    if PROM_AVAILABLE:
        JOB_COUNTER.labels(status=status).inc()


def track_job_duration(endpoint: str, duration_seconds: float) -> None:
    """Record an observation in ``quantx_job_duration_seconds``."""
    if PROM_AVAILABLE:
        JOB_DURATION.labels(endpoint=endpoint).observe(duration_seconds)


def update_cache_size(size: int) -> None:
    """Set ``quantx_cache_size`` gauge to *size*."""
    if PROM_AVAILABLE:
        CACHE_SIZE.set(size)


def update_jobs_active(delta: int) -> None:
    """Increment or decrement the ``quantx_jobs_active`` gauge by *delta*."""
    if PROM_AVAILABLE:
        JOBS_ACTIVE.inc(delta) if delta > 0 else JOBS_ACTIVE.dec(-delta)


# ---------------------------------------------------------------------------
# Starlette middleware — records per-request counter and duration
# ---------------------------------------------------------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    """Record ``quantx_requests_total`` and ``quantx_request_duration_seconds``
    for every HTTP request.

    The ``/metrics`` scrape endpoint itself is excluded to avoid recursion.
    """

    async def dispatch(
        self, request: StarletteRequest, call_next: Any
    ) -> StarletteResponse:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        if PROM_AVAILABLE and request.url.path != "/metrics":
            # Use the route template (e.g. "/jobs/{job_id}/status") rather than
            # the raw path to avoid cardinality explosion from UUIDs / ids.
            route = request.scope.get("route")
            endpoint = route.path if route is not None else request.url.path
            status = str(response.status_code)
            REQUEST_COUNTER.labels(endpoint=endpoint, status=status).inc()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)

        return response


# ---------------------------------------------------------------------------
# /metrics scrape endpoint
# ---------------------------------------------------------------------------

metrics_router = APIRouter(tags=["observability"])


@metrics_router.get("/metrics", include_in_schema=False)
async def scrape_metrics() -> Response:
    """Return Prometheus text-format metrics for scraping.

    Returns HTTP 503 when ``prometheus_client`` is not installed so that
    Prometheus scrape jobs can detect misconfigured targets.
    """
    if not PROM_AVAILABLE:
        return Response(
            content="# prometheus_client not installed\n",
            status_code=503,
            media_type="text/plain; charset=utf-8",
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
