"""Unit tests for tests/eval/faithfulness.py (P3.5).

Uses httpx.MockTransport to stub both chat-completion passes (extract + grade).
No network, no live chat model.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from tests.eval.faithfulness import (
    extract_claims,
    faithfulness,
    grade_claim,
    _parse_claim_lines,
    _parse_yes_no,
)


CHAT_URL = "http://fake-vllm:8000/v1"
CHAT_MODEL = "orgchat-chat"


def _chat_response(content: str, status: int = 200) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": "boom"})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


def _router(handlers: list[Callable[[httpx.Request], httpx.Response]]) -> httpx.MockTransport:
    """Return an httpx.MockTransport that dispatches successive calls to ``handlers``.

    Call N goes to handlers[N]; if the list runs out we fail the test loudly.
    """
    state = {"i": 0}

    def dispatch(req: httpx.Request) -> httpx.Response:
        i = state["i"]
        state["i"] += 1
        if i >= len(handlers):
            raise AssertionError(
                f"unexpected {i + 1}th request (only {len(handlers)} handlers wired)"
            )
        return handlers[i](req)

    return httpx.MockTransport(dispatch)


def _fixed(content: str, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that always returns the same chat response."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response(content, status=status)

    return handler


# ---------------------------------------------------------------------------
# Pure-function parsers
# ---------------------------------------------------------------------------


def test_parse_claim_lines_handles_dash_bullet():
    raw = "- the sky is blue\n- grass is green"
    assert _parse_claim_lines(raw) == ["the sky is blue", "grass is green"]


def test_parse_claim_lines_handles_numbered_list():
    raw = "1. the sky is blue\n2) grass is green\n3. water is wet"
    assert _parse_claim_lines(raw) == [
        "the sky is blue",
        "grass is green",
        "water is wet",
    ]


def test_parse_claim_lines_strips_surrounding_quotes():
    raw = '- "the sky is blue"'
    assert _parse_claim_lines(raw) == ["the sky is blue"]


def test_parse_claim_lines_dedupes_case_insensitively():
    raw = "- The sky is blue\n- the sky is blue\n- THE SKY IS BLUE"
    out = _parse_claim_lines(raw)
    assert len(out) == 1


def test_parse_claim_lines_caps_at_max_claims():
    lines = "\n".join(f"- claim {i}" for i in range(25))
    out = _parse_claim_lines(lines)
    assert len(out) == 10  # _MAX_CLAIMS


def test_parse_claim_lines_empty_or_whitespace_returns_empty():
    assert _parse_claim_lines("") == []
    assert _parse_claim_lines("   \n   \n") == []


def test_parse_yes_no_yes_variants():
    assert _parse_yes_no("YES") is True
    assert _parse_yes_no("yes") is True
    assert _parse_yes_no("Yes.") is True
    assert _parse_yes_no("YES, clearly supported") is True
    assert _parse_yes_no("yes\nbecause the context says so") is True


def test_parse_yes_no_no_variants():
    assert _parse_yes_no("NO") is False
    assert _parse_yes_no("no") is False
    assert _parse_yes_no("No, the claim is not supported.") is False


def test_parse_yes_no_ambiguous_counts_as_no():
    # Safety bias: anything non-YES is NO.
    assert _parse_yes_no("maybe") is False
    assert _parse_yes_no("probably") is False
    assert _parse_yes_no("unclear") is False
    assert _parse_yes_no("") is False
    assert _parse_yes_no("   ") is False
    assert _parse_yes_no("PROBABLY YES") is False  # leading token is 'probably'


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


async def test_extract_claims_returns_list():
    """'The sky is blue. Grass is green.' -> two claims."""
    transport = _router([_fixed("- the sky is blue\n- grass is green")])
    claims = await extract_claims(
        "The sky is blue. Grass is green.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert claims == ["the sky is blue", "grass is green"]


async def test_extract_claims_empty_answer_short_circuits():
    """Empty answer -> no network call, empty list."""

    def explode(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be hit for empty answer")

    claims = await extract_claims(
        "", chat_url=CHAT_URL, chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(explode),
    )
    assert claims == []


async def test_extract_claims_http_5xx_returns_empty():
    transport = _router([_fixed("", status=503)])
    claims = await extract_claims(
        "some answer",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert claims == []


async def test_extract_claims_malformed_json_returns_empty():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    claims = await extract_claims(
        "some answer",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert claims == []


async def test_extract_claims_sends_bearer_token_when_provided():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("- one")

    await extract_claims(
        "answer",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-secret",
        transport=httpx.MockTransport(handler),
    )
    assert seen["auth"] == "Bearer sk-secret"


# ---------------------------------------------------------------------------
# grade_claim
# ---------------------------------------------------------------------------


async def test_grade_claim_supported_returns_true():
    transport = _router([_fixed("YES")])
    ok = await grade_claim(
        context="the sky is blue",
        claim="sky is blue",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert ok is True


async def test_grade_claim_unsupported_returns_false():
    transport = _router([_fixed("NO")])
    ok = await grade_claim(
        context="the sky is blue",
        claim="the moon is made of cheese",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert ok is False


async def test_grade_claim_ambiguous_counts_as_false():
    transport = _router([_fixed("maybe")])
    ok = await grade_claim(
        context="the sky is blue",
        claim="the sun rises in the east",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert ok is False


async def test_grade_claim_strips_prefix_yes_comma_clearly():
    transport = _router([_fixed("YES, clearly supported by the context.")])
    ok = await grade_claim(
        context="ctx",
        claim="claim",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert ok is True


async def test_grade_claim_http_error_falls_to_false():
    transport = _router([_fixed("", status=500)])
    ok = await grade_claim(
        context="ctx",
        claim="claim",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# faithfulness (end-to-end)
# ---------------------------------------------------------------------------


async def test_faithfulness_fully_supported_scores_one():
    """Two claims, both graded YES -> score = 1.0."""
    transport = _router(
        [
            _fixed("- the sky is blue\n- grass is green"),
            _fixed("YES"),  # claim 1
            _fixed("YES"),  # claim 2
        ]
    )
    out = await faithfulness(
        context="the sky is blue and grass is green",
        answer="The sky is blue. Grass is green.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert out["score"] == 1.0
    assert out["n_claims"] == 2
    assert out["n_supported"] == 2
    assert out["unsupported"] == []


async def test_faithfulness_half_supported_scores_half():
    """Two claims, one YES one NO -> score = 0.5."""
    transport = _router(
        [
            _fixed("- the sky is blue\n- the moon is cheese"),
            _fixed("YES"),  # claim 1
            _fixed("NO"),   # claim 2
        ]
    )
    out = await faithfulness(
        context="the sky is blue",
        answer="The sky is blue. The moon is cheese.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert out["score"] == 0.5
    assert out["n_claims"] == 2
    assert out["n_supported"] == 1
    assert out["unsupported"] == ["the moon is cheese"]


async def test_faithfulness_empty_answer_is_vacuously_faithful():
    """Empty answer -> score 1.0, no network calls."""

    def explode(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be hit for empty answer")

    out = await faithfulness(
        context="the sky is blue",
        answer="",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(explode),
    )
    assert out["score"] == 1.0
    assert out["n_claims"] == 0
    assert out["n_supported"] == 0
    assert out["claims"] == []


async def test_faithfulness_whitespace_only_answer_is_vacuous():
    def explode(req: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be hit")

    out = await faithfulness(
        context="ctx",
        answer="   \n   \t  ",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(explode),
    )
    assert out["score"] == 1.0
    assert out["n_claims"] == 0


async def test_faithfulness_no_claims_extracted_is_vacuous():
    """Model extracts nothing -> score 1.0, no grading calls made."""
    # Only one handler wired: the extract pass. A second call would error.
    transport = _router([_fixed("")])
    out = await faithfulness(
        context="ctx",
        answer="Some answer",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert out["score"] == 1.0
    assert out["n_claims"] == 0


async def test_faithfulness_ambiguous_judge_counts_as_no():
    """Judge says 'maybe' -> claim counted as unsupported."""
    transport = _router(
        [
            _fixed("- the sky is blue"),
            _fixed("maybe"),  # ambiguous -> NO
        ]
    )
    out = await faithfulness(
        context="the sky is blue",
        answer="The sky is blue.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert out["score"] == 0.0
    assert out["n_claims"] == 1
    assert out["n_supported"] == 0
    assert out["unsupported"] == ["the sky is blue"]


async def test_faithfulness_yes_comma_clearly_treated_as_yes():
    """Judge says 'YES, clearly' -> claim counted as supported."""
    transport = _router(
        [
            _fixed("- the sky is blue"),
            _fixed("YES, clearly supported"),
        ]
    )
    out = await faithfulness(
        context="the sky is blue",
        answer="The sky is blue.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=transport,
    )
    assert out["score"] == 1.0
    assert out["n_supported"] == 1


async def test_faithfulness_sends_correct_endpoint_and_model():
    """First call hits /chat/completions with the right model."""
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(req.url),
            "body": json.loads(req.content.decode()),
        })
        # Return a claim on first call, YES on the second.
        if len(seen) == 1:
            return _chat_response("- claim one")
        return _chat_response("YES")

    out = await faithfulness(
        context="ctx",
        answer="Answer text.",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=httpx.MockTransport(handler),
    )
    assert out["score"] == 1.0
    assert len(seen) == 2
    for call in seen:
        assert call["url"].endswith("/chat/completions")
        assert call["body"]["model"] == CHAT_MODEL
