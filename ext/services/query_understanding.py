"""Query Understanding LLM — JSON-schema-guided intent + resolution.

Plan B Phase 4. This module is the LLM half of the hybrid regex+LLM router.
The fast regex path lives in :mod:`ext.services.query_intent`; this module is
invoked only when the router escalates (see Task 4.4 in Plan B).

Output is a constrained JSON object enforced by xgrammar guided_json on the
vLLM server side. We send the schema in the request; vLLM constrains
generation to valid output, eliminating the schema-violation noise that
plagued unconstrained LLM routers.

Schema::

    {
      "intent": "metadata" | "global" | "specific" | "specific_date",
      "resolved_query": "<rewritten standalone query, no pronouns>",
      "temporal_constraint": {
        "year": int | null, "quarter": int | null, "month": int | null
      } | null,
      "entities": ["<named entity>", ...],
      "confidence": float in [0.0, 1.0]
    }

The model is ``cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`` served by ``vllm-qu``
on GPU 1. This module is async and uses :mod:`httpx`; it has a hard deadline
(``RAG_QU_LATENCY_BUDGET_MS``, default 600 ms). On deadline-miss or HTTP
failure the caller falls back to regex-only classification — ``analyze_query``
returns ``None`` rather than raising.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


log = logging.getLogger("orgchat.qu")


# JSON schema fed to vLLM's guided_json. Keep this in lockstep with
# :class:`QueryUnderstanding` below — tests in
# ``test_query_understanding_schema.py`` enforce the shape.
#
# Phase 2.1 update: ``temporal_constraint`` now carries ``months`` and
# ``years`` arrays alongside the legacy singular ``year``/``quarter``/
# ``month`` fields. The LLM is instructed (see _PROMPT_TEMPLATE) to use
# the arrays whenever the query mentions multiple months or years
# ("activities in May, July, December and February 2023" → months:[2,5,7,12],
# years:[2023]). Single-month queries may use either form. Readers
# (retriever._filter_by_temporal_constraint) prefer the arrays when
# non-empty and fall back to the singular fields otherwise, so existing
# single-month behaviour is byte-equivalent.
QU_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "resolved_query",
        "temporal_constraint",
        "entities",
        "confidence",
    ],
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["metadata", "global", "specific", "specific_date"],
        },
        "resolved_query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "temporal_constraint": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "year", "quarter", "month", "months", "years",
                    ],
                    "properties": {
                        "year": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1900, "maximum": 2100},
                                {"type": "null"},
                            ],
                        },
                        "quarter": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1, "maximum": 4},
                                {"type": "null"},
                            ],
                        },
                        "month": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1, "maximum": 12},
                                {"type": "null"},
                            ],
                        },
                        "months": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 12,
                            },
                            "maxItems": 12,
                        },
                        "years": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 1900,
                                "maximum": 2100,
                            },
                            "maxItems": 20,
                        },
                    },
                },
            ],
        },
        "entities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 10,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}


@dataclass
class QueryUnderstanding:
    """Structured output from the QU LLM.

    Mirrors :data:`QU_OUTPUT_SCHEMA` plus two provenance fields the bridge
    uses for metric labelling and shadow-mode logging:

    - ``source`` is one of ``"llm"``, ``"regex"``, or ``"cached"`` once the
      result reaches the caller. Defaults to ``"llm"`` because this dataclass
      is constructed from the LLM response.
    - ``cached`` is set to ``True`` by the cache layer (Phase 4.5) so the
      bridge can distinguish hot-cache hits from cold LLM calls.
    """

    intent: str
    resolved_query: str
    temporal_constraint: Optional[dict]
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "llm"
    cached: bool = False


_VALID_INTENTS = {"metadata", "global", "specific", "specific_date"}


_PROMPT_TEMPLATE = """You are a query classifier and rewriter for a retrieval system.

Today's date is {today}.

Recent conversation context (oldest first):
{history_block}

Current user query: "{query}"

Your task:
1. Classify the query into ONE of: metadata, global, specific, specific_date.
   - metadata: enumeration / catalog questions ("list documents", "what files do I have")
   - global: aggregation / coverage across the corpus ("compare", "trends", "summarize all")
   - specific: single-document or content-anchored question
   - specific_date: question pinpointing one or more dates / months / quarters / years
2. Resolve the query into a standalone form. Replace pronouns ("it", "that") with
   their antecedents from history. Replace relative time ("last quarter", "yesterday")
   with absolute dates relative to today.
3. Extract a temporal_constraint object. Populate fields based on what the query says:
   - "months": list of month numbers (1-12) when ANY months are mentioned. Use this
     for multi-month queries: "May, July, December and February 2023" -> months: [2,5,7,12].
     For a single month, "months" may have one element OR set "month" alone.
   - "years": list of year numbers (1900-2100) when ANY years are mentioned. Same
     pattern: multi-year goes in the array, single-year may use either form.
   - "month", "quarter", "year": singular legacy fields. Set to null if the array
     form is populated, OR if no temporal anchor is mentioned. Quarter is 1-4 (Q1=Jan-Mar).
   - When the query has NO temporal anchor at all, set temporal_constraint = null.
   - When "since 2020", "from 2021 to 2023", or similar ranges appear, expand into
     the full years list (e.g. "2021 to 2023" -> years: [2021, 2022, 2023]).
   - Always include both "months" and "years" arrays in the object (use [] when
     not applicable) — required by the schema.
   Multi-temporal queries are common ("activities in May, Jul, Dec, Feb 2023",
   "compare Q1 and Q3 of 2023") — prefer the array form for those.
4. List the named entities (products, places, people, units, brigades) referenced.
5. Output your confidence in [0.0, 1.0].

Examples of temporal_constraint shape:
  - "What happened in February 2023?" ->
    {{"year": 2023, "quarter": null, "month": 2, "months": [], "years": []}}
  - "Activities in May, July, Dec, Feb 2023" ->
    {{"year": null, "quarter": null, "month": null,
      "months": [2,5,7,12], "years": [2023]}}
  - "Compare 2022 and 2023" ->
    {{"year": null, "quarter": null, "month": null,
      "months": [], "years": [2022, 2023]}}
  - "Summarize the entire corpus" ->
    null

Respond with JSON ONLY, conforming to the provided schema."""


def build_qu_prompt(query: str, history: list[dict]) -> str:
    """Compose the prompt for the QU LLM.

    ``history`` is a list of ``{role, content}`` dicts in chronological order
    (oldest first). We include up to the last 6 messages (3 turns), truncated
    to 200 chars each, to keep the prompt under the model's prefix-cache
    window.
    """
    today = _dt.date.today().isoformat()
    if history:
        truncated = history[-6:]
        lines = []
        for msg in truncated:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").replace("\n", " ").strip()
            if len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"  [{role}]: {content}")
        history_block = "\n".join(lines)
    else:
        history_block = "  (no previous turns)"
    return _PROMPT_TEMPLATE.format(
        today=today, history_block=history_block, query=query
    )


def parse_qu_response(raw: str) -> QueryUnderstanding:
    """Parse a JSON response from the QU LLM into a QueryUnderstanding.

    Raises :exc:`ValueError` on malformed JSON or invalid intent. Clamps
    ``confidence`` to ``[0.0, 1.0]`` silently — defensive, since the schema
    bounds it but pre-V1 vLLM occasionally violated guided_json bounds. Trims
    ``entities`` to the first 10 entries to mirror the schema.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"QU response not valid JSON: {e}") from e

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        raise ValueError(f"invalid intent in QU response: {intent!r}")

    resolved = data.get("resolved_query")
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError("resolved_query missing or empty")

    tc = data.get("temporal_constraint")
    if tc is not None and not isinstance(tc, dict):
        raise ValueError("temporal_constraint must be null or object")
    # Phase 2.1 — soft-validate the new arrays. The schema enforces the
    # bounds on the LLM side; this is defense-in-depth for the
    # occasional pre-V1 guided_json escape. Coerce to empty list on any
    # type drift so downstream filter helpers don't have to special-case.
    if isinstance(tc, dict):
        for k in ("months", "years"):
            v = tc.get(k)
            if v is None:
                tc[k] = []
                continue
            if not isinstance(v, list):
                tc[k] = []
                continue
            tc[k] = [int(x) for x in v if isinstance(x, (int, float))]
        # Cap months to 1..12 / years to 1900..2100 — the schema says so
        # but a non-strict guided_json can land out-of-bounds values.
        tc["months"] = [m for m in tc["months"] if 1 <= m <= 12]
        tc["years"] = [y for y in tc["years"] if 1900 <= y <= 2100]

    entities = data.get("entities") or []
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"confidence must be numeric: {e}") from e
    confidence = max(0.0, min(1.0, confidence))

    return QueryUnderstanding(
        intent=intent,
        resolved_query=resolved.strip(),
        temporal_constraint=tc,
        entities=[str(e) for e in entities[:10]],
        confidence=confidence,
        source="llm",
        cached=False,
    )


async def analyze_query(
    query: str,
    history: list[dict] | None = None,
    *,
    qu_url: str | None = None,
    model: str | None = None,
    timeout_ms: int | None = None,
) -> Optional[QueryUnderstanding]:
    """Call the QU LLM and return a parsed :class:`QueryUnderstanding`.

    Returns ``None`` on any failure — the caller falls back to regex.
    Soft-deadline via ``timeout_ms`` (default ``RAG_QU_LATENCY_BUDGET_MS``,
    600 ms); deadline misses are logged at WARN. The latency histogram
    (`rag_qu_latency_seconds`) is observed on every call regardless of
    outcome, and schema violations bump `rag_qu_schema_violations_total`.
    """
    import time as _time

    # Local imports keep the regex hot path (which never reaches here)
    # free of metrics import cost.
    from .metrics import rag_qu_latency, rag_qu_schema_violations

    qu_url = qu_url or os.environ.get("RAG_QU_URL", "http://vllm-qu:8000/v1")
    model = model or os.environ.get("RAG_QU_MODEL", "qwen3-4b-qu")
    if timeout_ms is None:
        timeout_ms = int(os.environ.get("RAG_QU_LATENCY_BUDGET_MS", "600"))
    timeout = max(timeout_ms / 1000.0, 0.001)

    prompt = build_qu_prompt(query=query, history=history or [])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output only JSON. No prose."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        # vLLM V1 supports the OpenAI-spec response_format with json_schema —
        # this is what xgrammar guided_json hooks into in practice. The
        # legacy ``extra_body.guided_json`` is silently dropped by vllm-
        # openai >= 0.6.4 (the model just emits arbitrary JSON instead).
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "query_understanding",
                "schema": QU_OUTPUT_SCHEMA,
                "strict": True,
            },
        },
    }

    # P3.1 — single retry on transient failures (timeout / connect /
    # read). Schema violations and HTTP 4xx do NOT retry.
    # Wave 2 (review §5.16): the original retry used HALF the per-call
    # timeout. On a saturated vllm-qu where the first 600ms attempt
    # times out, retrying with 300ms virtually guarantees a second
    # timeout and a regex fallback. Use the FULL timeout on retry —
    # total wall-clock is now ~2×timeout_ms but the second attempt
    # actually has a chance. RAG_QU_RETRY_TIMEOUT_MS overrides if the
    # operator wants asymmetric behaviour.
    _t0 = _time.monotonic()
    _last_exc: Exception | None = None
    _retry_timeout_env = os.environ.get("RAG_QU_RETRY_TIMEOUT_MS")
    if _retry_timeout_env:
        try:
            _retry_timeout = max(int(_retry_timeout_env) / 1000.0, 0.001)
        except (TypeError, ValueError):
            _retry_timeout = timeout
    else:
        _retry_timeout = timeout
    for _attempt in (1, 2):
        try:
            _per_attempt_timeout = timeout if _attempt == 1 else _retry_timeout
            async with httpx.AsyncClient(timeout=_per_attempt_timeout) as client:
                r = await client.post(f"{qu_url}/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
                raw = data["choices"][0]["message"]["content"]
                try:
                    return parse_qu_response(raw)
                except ValueError:
                    # Schema violation — bump the dedicated counter so the
                    # alert in observability/prometheus/alerts-qu.yml fires.
                    try:
                        rag_qu_schema_violations.inc()
                    except Exception:
                        pass
                    raise
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            _last_exc = e
            if _attempt == 1:
                log.info(
                    "QU LLM timed out after %dms (attempt 1/2) — retrying",
                    int(_per_attempt_timeout * 1000),
                )
                continue
            log.warning(
                "QU LLM timed out twice (budget %dms); falling back to regex",
                timeout_ms,
            )
        except (httpx.ConnectError, httpx.ReadError) as e:
            _last_exc = e
            if _attempt == 1:
                log.info(
                    "QU LLM transport error (attempt 1/2): %s — retrying", e,
                )
                continue
            log.warning("QU LLM transport error twice: %s; falling back", e)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("QU LLM error: %s; falling back to regex", e)
            try:
                rag_qu_latency.observe(_time.monotonic() - _t0)
            except Exception:
                pass
            return None
        finally:
            if _attempt == 2 or _last_exc is None:
                try:
                    rag_qu_latency.observe(_time.monotonic() - _t0)
                except Exception:
                    pass
    return None


__all__ = [
    "QueryUnderstanding",
    "QU_OUTPUT_SCHEMA",
    "analyze_query",
    "build_qu_prompt",
    "parse_qu_response",
]
