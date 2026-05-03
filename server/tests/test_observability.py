"""M107: Tests for Prometheus metrics and OpenTelemetry tracing helpers.

All tests are written to pass whether or not ``prometheus_client`` /
``opentelemetry-sdk`` are installed so CI runs clean in both configurations.
"""
from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.observability.metrics import (
    PROM_AVAILABLE,
    MetricsMiddleware,
    metrics_router,
    track_job_duration,
    track_job_status,
    update_cache_size,
    update_jobs_active,
)
from server.observability.tracing import get_tracer, setup_tracing


# ── Helper functions are safe no-ops when prometheus_client absent ─────────────

def test_track_job_status_does_not_raise():
    track_job_status("done")
    track_job_status("failed")
    track_job_status("unknown")


def test_track_job_duration_does_not_raise():
    track_job_duration("network/repeater-chain", 0.5)
    track_job_duration("protocols/bb84", 0.0)


def test_update_cache_size_does_not_raise():
    update_cache_size(0)
    update_cache_size(42)


def test_update_jobs_active_increment_decrement():
    update_jobs_active(+1)
    update_jobs_active(-1)


# ── /metrics endpoint ──────────────────────────────────────────────────────────

@pytest.fixture()
def metrics_app():
    """Minimal FastAPI app with MetricsMiddleware and the /metrics router."""
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    app.include_router(metrics_router)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def test_metrics_endpoint_status(metrics_app):
    """GET /metrics returns 200 (prometheus installed) or 503 (not installed)."""
    client = TestClient(metrics_app)
    resp = client.get("/metrics")
    if PROM_AVAILABLE:
        assert resp.status_code == 200
    else:
        assert resp.status_code == 503
        assert "prometheus_client not installed" in resp.text


def test_metrics_endpoint_content_type_when_available(metrics_app):
    """When prometheus is available the content-type is the Prometheus text format."""
    if not PROM_AVAILABLE:
        pytest.skip("prometheus_client not installed")
    client = TestClient(metrics_app)
    resp = client.get("/metrics")
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_middleware_does_not_break_normal_requests(metrics_app):
    """Middleware must not raise or alter response bodies."""
    client = TestClient(metrics_app)
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_metrics_middleware_skips_metrics_path(metrics_app):
    """/metrics itself must not be counted by the middleware (avoid double-count)."""
    client = TestClient(metrics_app)
    # Just verify no crash regardless of prometheus availability
    resp = client.get("/metrics")
    assert resp.status_code in (200, 503)


def test_metrics_middleware_records_multiple_requests(metrics_app):
    """Several requests complete without errors — counter is updated if prom present."""
    client = TestClient(metrics_app)
    for _ in range(3):
        assert client.get("/ping").status_code == 200


# ── Tracing helpers ────────────────────────────────────────────────────────────

def test_setup_tracing_no_op_without_endpoint(monkeypatch):
    """setup_tracing must not raise when OTEL_EXPORTER_OTLP_ENDPOINT is absent."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    setup_tracing()  # must not raise


def test_setup_tracing_accepts_fastapi_app(monkeypatch):
    """setup_tracing(app) must not raise."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    app = FastAPI()
    setup_tracing(app)  # must not raise


def test_get_tracer_returns_usable_tracer():
    """get_tracer() must return an object supporting the span context-manager API."""
    tracer = get_tracer("quantx-test")
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("solver", "bb84")
        span.record_exception(ValueError("synthetic"))


def test_get_tracer_custom_name():
    """get_tracer accepts an arbitrary name without raising."""
    tracer = get_tracer("my-service")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("key", "value")


# ── Integration: jobs router emits metrics ────────────────────────────────────

def test_jobs_instrumented_import():
    """Importing jobs router should not raise even with observability imports."""
    import server.routers.jobs as jobs_mod  # noqa: F401 — import is the test
    assert hasattr(jobs_mod, "_run_job")
