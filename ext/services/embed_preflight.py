"""Startup preflight: assert ``EMBED_MODEL`` is what TEI is actually serving.

Bug-fix campaign §3.6. Mirrors :func:`ext.services.chat_model_preflight.preflight_chat_model`
in spirit — runs once at app boot, validates a critical env-vs-runtime
contract, and surfaces mismatches in metrics so dashboards / alerts can
catch silent misconfigurations.

TEI exposes ``GET /info`` (NOT ``/v1/models``) which returns a JSON object
with ``model_id`` (the served model). We compare that against ``EMBED_MODEL``
in env and bump ``embed_model_mismatch_total`` + WARNING on miss.

Why warn instead of crash: operators sometimes register transparent model
aliases at the gateway level. A hard crash would prevent open-webui from
booting in those legitimate cases. A WARNING + counter increment
surfaces real mismatches (e.g. EMBED_MODEL was rolled back but TEI is
still on the new image) without breaking startup.

Why fail-open on endpoint unreachable: TEI cold-start can take ~30s;
preflight commonly runs before the endpoint is ready. Crashing here would
create a chicken-and-egg deadlock between open-webui and TEI. Better to
log and let the first embed call surface the real failure.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from ext.services.metrics import embed_model_mismatch_total

log = logging.getLogger("orgchat.preflight.embed_model")


def preflight_embedder(
    *,
    tei_url: Optional[str] = None,
    embed_model: Optional[str] = None,
    timeout: float = 5.0,
    transport: Optional[httpx.BaseTransport] = None,
) -> None:
    """Validate ``EMBED_MODEL`` against TEI's ``/info`` endpoint.

    Args:
        tei_url: optional override; defaults to ``TEI_URL`` env then
            an internal default (``http://tei:80``). Trailing ``/`` is
            stripped.
        embed_model: optional override; defaults to ``EMBED_MODEL`` env.
            When unset, the preflight returns cleanly (nothing to check).
        timeout: HTTP timeout in seconds for the ``/info`` GET.
        transport: optional :class:`httpx.BaseTransport`, primarily for tests.

    On mismatch (``model_id != EMBED_MODEL``):
      - ``embed_model_mismatch_total`` counter incremented.
      - WARNING logged with ``EMBED_MODEL`` and the served model_id.
      - returns cleanly (no exception).

    On endpoint failure (timeout / 5xx / malformed JSON):
      - WARNING logged.
      - returns cleanly. Does NOT bump the mismatch counter — the
        endpoint may be cold-starting; the counter is reserved for
        confirmed mismatches against a healthy endpoint.
    """
    embed_model = embed_model or os.environ.get("EMBED_MODEL")
    if not embed_model:
        log.info("embed-model preflight: EMBED_MODEL unset — skipping")
        return
    tei_url = tei_url or os.environ.get("TEI_URL") or "http://tei:80"
    tei_url = tei_url.rstrip("/")
    url = f"{tei_url}/info"
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        log.warning(
            "embed-model preflight: %s GET %s failed (%s) — "
            "skipping mismatch check (endpoint may be cold-starting). "
            "First embed call will surface the real failure if any.",
            type(exc).__name__,
            url,
            exc,
        )
        return

    served = data.get("model_id") if isinstance(data, dict) else None
    if not isinstance(served, str):
        log.warning(
            "embed-model preflight: %s returned no string model_id "
            "(got %r) — cannot validate. Skipping.",
            url, type(served).__name__,
        )
        return

    if served == embed_model:
        log.info(
            "embed-model preflight: EMBED_MODEL=%r confirmed on %s",
            embed_model, tei_url,
        )
        return

    # Mismatch — bump counter + WARNING. Do not crash; aliases may
    # legitimately rewrite the model name at the gateway.
    try:
        embed_model_mismatch_total.inc()
    except Exception:  # pragma: no cover — metrics is fail-open
        pass
    log.warning(
        "embed-model preflight: EMBED_MODEL=%r != served model_id=%r on %s. "
        "Operators may have aliased the model name; if not, similarity-score "
        "interpretation downstream (rerank thresholds, MMR lambda) is "
        "miscalibrated. Bumped embed_model_mismatch_total.",
        embed_model, served, tei_url,
    )


__all__ = ["preflight_embedder"]
