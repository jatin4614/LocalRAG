"""OpenTelemetry bootstrap for the OrgChat FastAPI extension.

Design invariants (read before editing):

1. **Zero OTel imports at module load.** When ``OBS_ENABLED`` is unset
   or "false", this module MUST NOT import ``opentelemetry.*``. All
   heavy imports live inside :func:`init_observability`. This lets the
   extension boot in environments where OTel is not installed (e.g.
   slim CI images, upstream tests, local dev).

2. **Fail-open everywhere.** Every public helper (``get_tracer``,
   ``get_meter``, ``span``, ``inject_context_into_headers``,
   ``extract_context_from_headers``) is safe to call when
   observability is disabled or when init failed. Callers never need
   to guard with ``if OBS_ENABLED``.

3. **Idempotent init.** Calling :func:`init_observability` twice is a
   no-op on the second call (guarded by a module-level flag).

4. **Additive to Prometheus.** This module does not touch the existing
   ``/metrics`` mount or :mod:`ext.services.metrics`. OTLP metric
   export is *additional*; Prometheus continues to work exactly as
   before.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

from .request_ctx import request_id_var, user_id_var

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state. All None until init succeeds.
# ---------------------------------------------------------------------------
_INITIALIZED: bool = False
_ENABLED: bool = False
_TRACER_PROVIDER: Any = None
_METER_PROVIDER: Any = None
_LOGGER_PROVIDER: Any = None
_PROPAGATOR: Any = None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _read_service_version() -> str:
    """Parse ``pyproject.toml`` for ``[project].version``.

    Falls back to ``0.0.0`` on any error — we'd rather emit spans with
    a wrong version than crash init.
    """
    try:
        import pathlib

        here = pathlib.Path(__file__).resolve()
        # ext/services/obs.py → repo root is parents[2]
        pyproject = here.parents[2] / "pyproject.toml"
        if not pyproject.exists():
            return "0.0.0"
        text = pyproject.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("version"):
                # crude parse: version = "0.1.0"
                _, _, rhs = line.partition("=")
                return rhs.strip().strip('"').strip("'") or "0.0.0"
    except Exception:  # pragma: no cover - defensive
        pass
    return "0.0.0"


# ---------------------------------------------------------------------------
# No-op fallbacks. Used when OBS is disabled or init failed.
# ---------------------------------------------------------------------------
class _NoopSpan:
    def set_attribute(self, *_a, **_kw) -> None: ...
    def set_attributes(self, *_a, **_kw) -> None: ...
    def record_exception(self, *_a, **_kw) -> None: ...
    def set_status(self, *_a, **_kw) -> None: ...
    def add_event(self, *_a, **_kw) -> None: ...
    def end(self, *_a, **_kw) -> None: ...
    def __enter__(self) -> "_NoopSpan":
        return self
    def __exit__(self, *_a) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, _name: str, *_a, **_kw) -> Iterator[_NoopSpan]:
        yield _NoopSpan()

    def start_span(self, *_a, **_kw) -> _NoopSpan:
        return _NoopSpan()


class _NoopCounter:
    def add(self, *_a, **_kw) -> None: ...


class _NoopHistogram:
    def record(self, *_a, **_kw) -> None: ...


class _NoopUpDownCounter:
    def add(self, *_a, **_kw) -> None: ...


class _NoopMeter:
    def create_counter(self, *_a, **_kw) -> _NoopCounter:
        return _NoopCounter()

    def create_histogram(self, *_a, **_kw) -> _NoopHistogram:
        return _NoopHistogram()

    def create_up_down_counter(self, *_a, **_kw) -> _NoopUpDownCounter:
        return _NoopUpDownCounter()

    def create_observable_gauge(self, *_a, **_kw) -> None:
        return None


_NOOP_TRACER = _NoopTracer()
_NOOP_METER = _NoopMeter()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_observability(app: Optional[FastAPI] = None) -> None:
    """Configure OTel exporters + instrumentors. Idempotent, fail-open.

    Reads ``OBS_ENABLED`` — when false/unset, returns immediately. When
    true but any import or setup step fails, logs a warning and leaves
    the module in the disabled state (public helpers keep returning
    no-ops).
    """
    global _INITIALIZED, _ENABLED
    global _TRACER_PROVIDER, _METER_PROVIDER, _LOGGER_PROVIDER, _PROPAGATOR

    if _INITIALIZED:
        # Still wire FastAPI instrumentation if a new app was provided
        # on a subsequent call and we are enabled.
        if _ENABLED and app is not None:
            _install_fastapi_instrumentation(app)
            _install_request_context_middleware(app)
        return
    _INITIALIZED = True

    if not _env_bool("OBS_ENABLED", False):
        _logger.info("observability disabled (OBS_ENABLED not set)")
        # Even when disabled we still install the request-context
        # middleware so ``request_id``/``user_id`` logging works.
        if app is not None:
            _install_request_context_middleware(app)
        return

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://orgchat-obs-otelcol:4317"
    )
    environment = os.getenv(
        "DEPLOYMENT_ENVIRONMENT", os.getenv("ENV", "development")
    )
    service_version = _read_service_version()

    try:
        # ------------------------------------------------------------------
        # Lazy imports — only when OBS is enabled.
        # ------------------------------------------------------------------
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", "orgchat-ext"),
                "service.namespace": "orgchat",
                "service.version": service_version,
                "deployment.environment": environment,
            }
        )

        # --- Traces ---
        tp = TracerProvider(resource=resource)
        tp.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(tp)
        _TRACER_PROVIDER = tp

        # --- Metrics ---
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=60_000,
        )
        mp = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(mp)
        _METER_PROVIDER = mp

        # --- Logs ---
        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=True))
        )
        set_logger_provider(lp)
        _LOGGER_PROVIDER = lp
        # Attach OTel logging handler to root so stdlib logs are exported.
        try:
            otel_handler = LoggingHandler(level=logging.NOTSET, logger_provider=lp)
            logging.getLogger().addHandler(otel_handler)
        except Exception as e:  # pragma: no cover - defensive
            _logger.warning("failed attaching OTel LoggingHandler: %s", e)

        _PROPAGATOR = TraceContextTextMapPropagator()
        _ENABLED = True

        _logger.info(
            "observability enabled: endpoint=%s env=%s version=%s",
            endpoint,
            environment,
            service_version,
        )
    except Exception as e:
        _logger.warning("observability init failed — running disabled: %s", e)
        _ENABLED = False
        # Still install request-context middleware for log enrichment.
        if app is not None:
            _install_request_context_middleware(app)
        return

    # ----- Auto-instrumentors (each guarded independently) -----
    if app is not None:
        _install_fastapi_instrumentation(app)

    _try_instrument("sqlalchemy", "opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor")
    _try_instrument("redis", "opentelemetry.instrumentation.redis", "RedisInstrumentor")
    _try_instrument("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor")
    _try_instrument("requests", "opentelemetry.instrumentation.requests", "RequestsInstrumentor")
    _try_instrument("psycopg2", "opentelemetry.instrumentation.psycopg2", "Psycopg2Instrumentor")
    _try_instrument("celery", "opentelemetry.instrumentation.celery", "CeleryInstrumentor")

    if app is not None:
        _install_request_context_middleware(app)


def _try_instrument(label: str, module_path: str, class_name: str) -> None:
    """Best-effort ``Instrumentor().instrument()`` call. Logs + swallows
    failures so a missing instrumentation package never breaks boot."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        cls().instrument()
        _logger.debug("instrumented %s", label)
    except Exception as e:
        _logger.warning("instrumentation %s skipped: %s", label, e)


def _install_fastapi_instrumentation(app: FastAPI) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _logger.debug("instrumented fastapi app")
    except Exception as e:
        _logger.warning("instrumentation fastapi skipped: %s", e)


def _install_request_context_middleware(app: FastAPI) -> None:
    """Assigns ``request_id`` / ``user_id`` contextvars for every request.

    Safe to install even when OBS is disabled — it just enriches logs.
    """
    import uuid

    # Guard against double-install on idempotent init.
    if getattr(app.state, "_orgchat_req_ctx_installed", False):
        return
    app.state._orgchat_req_ctx_installed = True

    @app.middleware("http")
    async def _request_context_middleware(request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        uid = request.headers.get("x-user-id")  # best-effort; auth may override later
        r_token = request_id_var.set(rid)
        u_token = user_id_var.set(uid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(r_token)
            user_id_var.reset(u_token)
        try:
            response.headers["x-request-id"] = rid
        except Exception:
            pass
        return response


def get_tracer(name: str = "orgchat") -> Any:
    """Return a tracer. No-op shim when OBS is disabled."""
    if not _ENABLED:
        return _NOOP_TRACER
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:
        return _NOOP_TRACER


def get_meter(name: str = "orgchat") -> Any:
    """Return a meter. No-op shim when OBS is disabled."""
    if not _ENABLED:
        return _NOOP_METER
    try:
        from opentelemetry import metrics

        return metrics.get_meter(name)
    except Exception:
        return _NOOP_METER


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Convenience context manager: start span, record exceptions, set status.

    Always safe to use — falls through to a no-op span when OBS is
    disabled so call sites never need to branch.
    """
    if not _ENABLED:
        yield _NoopSpan()
        return

    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = trace.get_tracer("orgchat")
    except Exception:
        yield _NoopSpan()
        return

    with tracer.start_as_current_span(name) as sp:
        try:
            if attrs:
                for k, v in attrs.items():
                    try:
                        sp.set_attribute(k, v)
                    except Exception:
                        pass
            yield sp
        except Exception as exc:
            try:
                sp.record_exception(exc)
                sp.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise


def inject_context_into_headers(headers: Optional[dict] = None) -> dict:
    """Inject W3C traceparent into ``headers`` for outbound calls.

    Returns the (possibly mutated) headers dict so call sites can do
    ``client.post(url, headers=obs.inject_context_into_headers({...}))``.
    """
    headers = dict(headers or {})
    if not _ENABLED or _PROPAGATOR is None:
        return headers
    try:
        _PROPAGATOR.inject(headers)
    except Exception:
        pass
    return headers


def extract_context_from_headers(headers: Optional[dict]) -> Any:
    """Extract a trace context from inbound ``headers``. Returns the
    opaque context object (or ``None`` when disabled).

    Usage::

        ctx = obs.extract_context_from_headers(request.headers)
        with tracer.start_as_current_span("task", context=ctx):
            ...
    """
    if not _ENABLED or _PROPAGATOR is None or not headers:
        return None
    try:
        return _PROPAGATOR.extract(dict(headers))
    except Exception:
        return None


def is_enabled() -> bool:
    """True when observability has been successfully initialized."""
    return _ENABLED
