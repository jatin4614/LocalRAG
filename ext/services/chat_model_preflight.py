"""Startup preflight: assert ``CHAT_MODEL`` is in chat endpoint /v1/models.

Part of bug-fix campaign §6.7. Mirrors :func:`ext.services.budget.preflight_tokenizer`
in spirit — runs once at app boot, validates a critical env-vs-runtime
contract, and surfaces mismatches in metrics so dashboards / alerts can
catch silent misconfigurations.

Why warn instead of crash: operators sometimes register transparent model
aliases on the chat endpoint (``model-manager`` rewrites are common in
this deploy). A hard crash would prevent open-webui from booting in
those legitimate cases. A WARNING + counter increment surfaces real
mismatches (e.g. ``CHAT_MODEL`` rolled back but ``vllm-chat`` is still
on the new build) without breaking startup.

Why fail-open on endpoint unreachable: ``vllm-chat`` cold-start can take
~60s; preflight commonly runs before the endpoint is ready. Crashing
here would create a chicken-and-egg deadlock between open-webui and
vllm-chat. Better to log and let the first chat request surface the
real failure.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from ext.services.metrics import chat_model_mismatch_total

log = logging.getLogger("orgchat.preflight.chat_model")


def preflight_chat_model(
    *,
    chat_url: Optional[str] = None,
    chat_model: Optional[str] = None,
    timeout: float = 5.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> None:
    """Validate ``CHAT_MODEL`` against the chat endpoint's ``/v1/models``.

    Args:
        chat_url: optional override; defaults to ``OPENAI_API_BASE_URL``
            then a sensible internal default. Trailing ``/`` stripped.
        chat_model: optional override; defaults to env ``CHAT_MODEL``.
            When unset, the preflight returns cleanly (nothing to check).
        timeout: HTTP timeout in seconds for the ``/v1/models`` GET.
        transport: optional :class:`httpx.BaseTransport`, primarily for tests.

    On mismatch (model not in the response ``data[].id`` list):
      - ``chat_model_mismatch_total`` counter incremented.
      - WARNING logged with ``CHAT_MODEL`` and the endpoint URL.
      - returns cleanly (no exception).

    On endpoint failure (timeout / 5xx / malformed JSON):
      - WARNING logged.
      - returns cleanly. Does NOT bump the mismatch counter — the
        endpoint may be cold-starting; the counter is reserved for
        confirmed mismatches against a healthy endpoint.
    """
    chat_model = chat_model or os.environ.get("CHAT_MODEL")
    if not chat_model:
        log.info("chat-model preflight: CHAT_MODEL unset — skipping")
        return
    chat_url = (
        chat_url
        or os.environ.get("OPENAI_API_BASE_URL")
        or "http://vllm-chat:8000/v1"
    )
    chat_url = chat_url.rstrip("/")
    url = f"{chat_url}/models"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        log.warning(
            "chat-model preflight: %s GET %s failed (%s) — "
            "skipping mismatch check (endpoint may be cold-starting). "
            "First chat request will surface the real failure if any.",
            type(exc).__name__,
            url,
            exc,
        )
        return

    served = []
    raw = data.get("data") if isinstance(data, dict) else None
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                ident = entry.get("id")
                if isinstance(ident, str):
                    served.append(ident)

    if chat_model in served:
        log.info(
            "chat-model preflight: CHAT_MODEL=%r confirmed on %s",
            chat_model,
            chat_url,
        )
        return

    # Mismatch — bump counter + WARNING. Do not crash; aliases may
    # legitimately rewrite the model name at the endpoint.
    try:
        chat_model_mismatch_total.inc()
    except Exception:  # pragma: no cover — metrics is fail-open
        pass
    log.warning(
        "chat-model preflight: CHAT_MODEL=%r not in /v1/models on %s "
        "(served=%s). Operators may have aliased the model name; if not, "
        "the next chat request will fail with 404 model_not_found. "
        "Bumped chat_model_mismatch_total.",
        chat_model,
        chat_url,
        served,
    )


__all__ = ["preflight_chat_model"]
