"""Per-request contextvars shared by logging + observability.

These are used by ``logging_setup`` to enrich log records and by the
request middleware in ``obs`` to carry ``request_id`` / ``user_id``
through async call stacks without threading them explicitly.

Kept in a dedicated module so callers can import them without pulling
in the full OTel bootstrap (which lazily imports heavy SDK packages).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
