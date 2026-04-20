"""RAGAS-style faithfulness: fraction of answer claims supported by context.

Two-pass LLM judge:
  1. Extract claims from answer -> list[str]
  2. For each claim, grade support given context -> bool
  score = supported / total. Unsupported claims list is also returned.

Edge cases:
  * Empty answer -> score 1.0 (vacuously faithful, no claims to be unfaithful about).
  * No claims extracted -> score 1.0 (same rationale).
  * Ambiguous judge response ("maybe", "probably") -> counted as NO (safety).
  * Bias note: when the same model generates the answer AND grades it, the judge
    can be systematically optimistic. Use a different/stronger judge when possible.

All network calls go through ``httpx.AsyncClient``. Tests inject
``httpx.MockTransport`` via the ``transport`` kwarg to avoid network.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx

log = logging.getLogger("orgchat.faithfulness")


_CLAIM_EXTRACT_PROMPT = """Break the following ANSWER into a flat list of standalone factual claims.

ANSWER:
{answer}

Output format: one claim per line, prefixed with "- ". No preamble. Maximum 10 claims. Each claim must be a complete, self-contained assertion."""


_CLAIM_GRADE_PROMPT = """Given the CONTEXT and a CLAIM, decide whether the claim is SUPPORTED by the context.

CONTEXT:
{context}

CLAIM:
{claim}

Answer with exactly one word: "YES" if the claim is supported (verbatim or by direct inference), or "NO" if the claim is not present in the context or contradicted.

Answer:"""


_MAX_CLAIMS = 10
_CLAIM_MAX_CHARS = 400
_CONTEXT_MAX_CHARS = 12_000  # keep prompts reasonable; scale with your chat model window


def _chat_url(base: str) -> str:
    return f"{base.rstrip('/')}/chat/completions"


def _auth_headers(api_key: Optional[str]) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _parse_claim_lines(raw: str) -> list[str]:
    """Parse ``- claim`` lines out of a raw chat response.

    Accepts ``- claim``, ``* claim``, numbered lists (``1. claim``) and plain
    newline-separated lines as a fallback. De-duplicates case-insensitively
    while preserving order. Caps at ``_MAX_CLAIMS`` for cost control.
    """
    lines: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        # Strip leading list markers: "- ", "* ", "1. ", "1) "
        m = re.match(r"^\s*(?:[-*]\s+|\d+[\.\)]\s+)(.*)$", s)
        if m:
            s = m.group(1).strip()
        # Drop surrounding quotes the model sometimes adds
        s = s.strip('"').strip("'").strip()
        if not s:
            continue
        if len(s) > _CLAIM_MAX_CHARS:
            s = s[:_CLAIM_MAX_CHARS]
        lines.append(s)

    seen: set[str] = set()
    deduped: list[str] = []
    for c in lines:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= _MAX_CLAIMS:
            break
    return deduped


def _parse_yes_no(raw: str) -> bool:
    """Interpret a judge response as YES/NO.

    Rules (in order):
      * Empty/whitespace -> NO (safety).
      * Case-insensitive leading token "yes" -> YES (handles "YES, clearly").
      * Case-insensitive leading token "no" -> NO (handles "No.").
      * Anything else ("maybe", "probably", "unclear") -> NO (safety).
    """
    if not raw:
        return False
    s = raw.strip().lower()
    if not s:
        return False
    # Match the leading word token only — "yes, clearly" / "yes." / "yes\nbecause..."
    m = re.match(r"([a-z]+)", s)
    if not m:
        return False
    head = m.group(1)
    if head == "yes":
        return True
    # "no" and everything else -> NO (deliberate safety bias).
    return False


async def extract_claims(
    answer: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout: float = 20.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> list[str]:
    """Ask the chat model to break ``answer`` into a flat list of claims.

    On any failure (network, 5xx, malformed JSON, empty content) returns ``[]``
    — callers treat that as "vacuously faithful" and score 1.0.
    """
    if not answer or not answer.strip():
        return []
    body = {
        "model": chat_model,
        "messages": [
            {"role": "user", "content": _CLAIM_EXTRACT_PROMPT.format(answer=answer.strip())},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            r = await client.post(_chat_url(chat_url), json=body, headers=_auth_headers(api_key))
            r.raise_for_status()
            data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open by design
        log.debug("claim extraction failed: %s", e)
        return []
    return _parse_claim_lines(content)


async def grade_claim(
    context: str,
    claim: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> bool:
    """Ask the chat model whether ``claim`` is supported by ``context``.

    Returns ``True`` only if the judge answers a clear YES. On any failure or
    ambiguous response, returns ``False`` (deliberate safety bias — we prefer
    to score unsupported rather than over-credit the answer).
    """
    if not claim or not claim.strip():
        return False
    ctx = (context or "").strip()
    if len(ctx) > _CONTEXT_MAX_CHARS:
        ctx = ctx[:_CONTEXT_MAX_CHARS]
    body = {
        "model": chat_model,
        "messages": [
            {
                "role": "user",
                "content": _CLAIM_GRADE_PROMPT.format(context=ctx, claim=claim.strip()),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            r = await client.post(_chat_url(chat_url), json=body, headers=_auth_headers(api_key))
            r.raise_for_status()
            data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:  # noqa: BLE001 — fail-open to NO
        log.debug("claim grading failed: %s", e)
        return False
    return _parse_yes_no(content)


async def faithfulness(
    context: str,
    answer: str,
    *,
    chat_url: str,
    chat_model: str,
    api_key: Optional[str] = None,
    extract_timeout: float = 20.0,
    grade_timeout: float = 10.0,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> dict:
    """Score the faithfulness of ``answer`` against ``context``.

    Returns a dict::

        {
            "score": float in [0.0, 1.0],
            "n_claims": int,
            "n_supported": int,
            "claims": list[str],
            "unsupported": list[str],
        }

    Vacuous cases (empty answer, zero claims extracted) score 1.0 with empty
    lists. This matches RAGAS' treatment of empty outputs as non-violating.
    """
    # Vacuous case: empty answer -> no claims to be unfaithful about.
    if not answer or not answer.strip():
        return {
            "score": 1.0,
            "n_claims": 0,
            "n_supported": 0,
            "claims": [],
            "unsupported": [],
        }

    claims = await extract_claims(
        answer,
        chat_url=chat_url,
        chat_model=chat_model,
        api_key=api_key,
        timeout=extract_timeout,
        transport=transport,
    )
    if not claims:
        # Model couldn't or wouldn't extract any claims — treat as vacuous.
        return {
            "score": 1.0,
            "n_claims": 0,
            "n_supported": 0,
            "claims": [],
            "unsupported": [],
        }

    supported: list[bool] = []
    unsupported: list[str] = []
    for claim in claims:
        ok = await grade_claim(
            context,
            claim,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            timeout=grade_timeout,
            transport=transport,
        )
        supported.append(ok)
        if not ok:
            unsupported.append(claim)

    n_claims = len(claims)
    n_supported = sum(1 for s in supported if s)
    score = n_supported / n_claims if n_claims else 1.0
    return {
        "score": score,
        "n_claims": n_claims,
        "n_supported": n_supported,
        "claims": claims,
        "unsupported": unsupported,
    }
