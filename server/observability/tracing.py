"""M107: OpenTelemetry tracing — no-op unless the OTel SDK is installed
and ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured.

Install (optional)::

    pip install opentelemetry-sdk opentelemetry-instrumentation-fastapi \\
                opentelemetry-exporter-otlp-proto-grpc

Usage::

    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 uvicorn server.main:app
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Optional OTel import
# ---------------------------------------------------------------------------

_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# No-op fallbacks (used when OTel is not installed)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def record_exception(self, exc: BaseException) -> None:  # noqa: ARG002
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:  # noqa: ARG002
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_tracing(app: "FastAPI | None" = None) -> None:
    """Initialise OpenTelemetry tracing.

    Does nothing if:
    - ``opentelemetry-sdk`` is not installed, OR
    - ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var is not set.

    This ensures zero overhead in deployments that haven't configured a
    collector.
    """
    if not _OTEL_AVAILABLE:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return  # No-op when no exporter configured

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        # grpc exporter not installed — skip silently
        return

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    _otel_trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass  # Instrumentation package not installed — skip


def get_tracer(name: str = "quantx") -> Any:
    """Return an OTel ``Tracer``, or a lightweight no-op if OTel is absent.

    Example::

        tracer = get_tracer()
        with tracer.start_as_current_span("solve") as span:
            span.set_attribute("solver", solver_name)
            result = engine.run_solver(solver_name, params)
    """
    if not _OTEL_AVAILABLE:
        return _NoOpTracer()
    return _otel_trace.get_tracer(name)
