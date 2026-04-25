"""Structured JSON logging configurator.

Attaches a JSON formatter to the root logger. Each record includes:

    timestamp, level, logger, message, trace_id, span_id, user_id, request_id

``trace_id`` / ``span_id`` are pulled from the current OTel context
when OBS is enabled — the extraction is wrapped in try/except so this
module works even when ``opentelemetry`` is not installed at all.

``user_id`` and ``request_id`` come from :mod:`ext.services.request_ctx`
(populated by the FastAPI middleware registered in :mod:`obs`).

Prefers ``python-json-logger`` when importable; falls back to a minimal
inline formatter otherwise.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

from .request_ctx import request_id_var, user_id_var

_CONFIGURED: bool = False


def _current_trace_ids() -> tuple[str, str]:
    """Return (trace_id_hex, span_id_hex). Empty strings when unavailable."""
    try:
        from opentelemetry import trace  # type: ignore

        sp = trace.get_current_span()
        if sp is None:
            return "", ""
        ctx = sp.get_span_context()
        if not ctx or not getattr(ctx, "is_valid", False):
            return "", ""
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:
        return "", ""


class _FallbackJsonFormatter(logging.Formatter):
    """Minimal stdlib-only JSON formatter used when python-json-logger
    isn't installed. Produces one JSON object per line."""

    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        tid, sid = _current_trace_ids()
        payload: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": tid,
            "span_id": sid,
            "request_id": request_id_var.get() or "",
            "user_id": user_id_var.get() or "",
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Preserve any structured extras passed via ``logger.info(..., extra={...})``.
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k.startswith("_"):
                continue
            if k in payload:
                continue
            try:
                json.dumps(v)  # test serializability
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        return json.dumps(payload, default=str)


class _ContextFilter(logging.Filter):
    """Inject trace/span/request/user into every record as attributes so
    ``python-json-logger`` picks them up via its ``rename_fields`` /
    attribute exposure path."""

    def filter(self, record: logging.LogRecord) -> bool:
        tid, sid = _current_trace_ids()
        record.trace_id = tid
        record.span_id = sid
        record.request_id = request_id_var.get() or ""
        record.user_id = user_id_var.get() or ""
        return True


def configure_json_logging(level: str = "INFO") -> None:
    """Install a JSON formatter on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    lvl_name = os.getenv("LOG_LEVEL", level).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)

    # Remove the default stderr handler if it's a plain one so we don't
    # emit duplicate records. Keep any non-stream handlers untouched.
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not getattr(
            h, "_orgchat_json", False
        ):
            root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler._orgchat_json = True  # type: ignore[attr-defined]
    handler.addFilter(_ContextFilter())

    formatter: logging.Formatter
    try:
        from pythonjsonlogger import jsonlogger  # type: ignore

        fmt = (
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "%(trace_id)s %(span_id)s %(request_id)s %(user_id)s"
        )
        formatter = jsonlogger.JsonFormatter(
            fmt,
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
            timestamp=True,
        )
    except Exception:
        formatter = _FallbackJsonFormatter()

    handler.setFormatter(formatter)
    root.addHandler(handler)
