"""Unit tests for ``ext.services.hyde`` (P3.3).

Covers the chat-generation primitives (``_generate_one``,
``generate_hypothetical_docs``), the embedding-averaging layer
(``hyde_embed``), and the flag helper (``is_enabled``).

No network is touched — ``httpx.MockTransport`` stubs the chat endpoint.
A tiny in-memory embedder implements the ``Embedder`` protocol so we can
assert exact mean-pooling math.
"""
from __future__ import annotations

import math

import httpx
import pytest

from ext.services import flags
from ext.services.hyde import (
    _generate_one,
    generate_hypothetical_docs,
    hyde_embed,
    is_enabled,
)


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    n = _norm(v) or 1.0
    return [x / n for x in v]


CHAT_URL = "http://fake-vllm:8000/v1"
CHAT_MODEL = "orgchat-chat"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transport(handler):
    """Wrap a request handler into an ``httpx.MockTransport``."""
    return httpx.MockTransport(handler)


def _chat_response(content: str, status: int = 200) -> httpx.Response:
    """OpenAI-compatible chat-completions response with ``content``."""
    if status != 200:
        return httpx.Response(status, json={"error": "server error"})
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


class _FixedEmbedder:
    """Returns a pre-programmed vector for each input text (by exact match)."""

    def __init__(self, mapping: dict[str, list[float]], default: list[float] | None = None) -> None:
        self._mapping = mapping
        self._default = default
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            if t in self._mapping:
                out.append(list(self._mapping[t]))
            elif self._default is not None:
                out.append(list(self._default))
            else:
                raise KeyError(f"no vector configured for {t!r}")
        return out


# ---------------------------------------------------------------------------
# _generate_one
# ---------------------------------------------------------------------------

async def test_generate_one_returns_excerpt_on_success():
    """Happy path: ``_generate_one`` returns the content verbatim (trimmed)."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = req.content.decode()
        return _chat_response("Refunds for enterprise tier customers are issued within 30 days.")

    out = await _generate_one(
        "refund policy for B2B",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-test",
        transport=_make_transport(handler),
    )
    assert out == "Refunds for enterprise tier customers are issued within 30 days."
    assert seen["url"].endswith("/chat/completions")
    # The query is embedded in the prompt.
    assert "refund policy for B2B" in seen["body"]


async def test_generate_one_returns_none_on_5xx():
    """HTTP 5xx → fail-open → None."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=503)

    out = await _generate_one(
        "anything",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None


async def test_generate_one_returns_none_on_timeout():
    """Transport-level timeout → caught → None."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("chat model too slow", request=req)

    out = await _generate_one(
        "anything",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None


async def test_generate_one_returns_none_on_empty_response():
    """Empty response body → None (nothing useful to embed)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("")

    out = await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None


async def test_generate_one_returns_none_when_response_too_long():
    """Response > 2000 chars → None (model ignored the length guard)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("x" * 2500)

    out = await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None


async def test_generate_one_strips_whitespace():
    """Leading/trailing whitespace is stripped from the response."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("   Policy excerpt.   \n")

    out = await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == "Policy excerpt."


async def test_generate_one_sends_bearer_token():
    """API key is forwarded as Authorization header."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("ok excerpt")

    await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key="sk-secret",
        transport=_make_transport(handler),
    )
    assert seen["auth"] == "Bearer sk-secret"


async def test_generate_one_omits_auth_when_no_api_key():
    """Without an API key, no Authorization header is sent."""

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        return _chat_response("excerpt")

    await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        api_key=None,
        transport=_make_transport(handler),
    )
    assert seen["auth"] is None


async def test_generate_one_malformed_json_falls_open():
    """Missing ``choices`` key → None (we don't crash on upstream bugs)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"totally_wrong": "shape"})

    out = await _generate_one(
        "q",
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None


# ---------------------------------------------------------------------------
# generate_hypothetical_docs
# ---------------------------------------------------------------------------

async def test_generate_n_of_3_all_succeed():
    """N=3 with a happy handler → list of 3."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("excerpt A")

    out = await generate_hypothetical_docs(
        "q",
        n=3,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert len(out) == 3
    assert all(x == "excerpt A" for x in out)


async def test_generate_n_drops_failures():
    """With some 500s mixed in, only successes come back."""

    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # Every other call returns 500.
        if calls["n"] % 2 == 0:
            return _chat_response("", status=500)
        return _chat_response(f"excerpt {calls['n']}")

    out = await generate_hypothetical_docs(
        "q",
        n=3,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    # Should return only the successful ones (≤ 3).
    assert 0 < len(out) < 3
    assert all(x.startswith("excerpt") for x in out)


async def test_generate_n_all_fail_returns_empty_list():
    """All generations fail → [] (caller will fall back)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=500)

    out = await generate_hypothetical_docs(
        "q",
        n=3,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out == []


async def test_generate_n_zero_returns_empty():
    """N=0 short-circuits with no chat calls."""

    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no chat call should have happened")

    out = await generate_hypothetical_docs(
        "q",
        n=0,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(should_not_be_called),
    )
    assert out == []


async def test_generate_n_negative_returns_empty():
    """Negative N is treated as zero — no panic."""

    def should_not_be_called(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no chat call should have happened")

    out = await generate_hypothetical_docs(
        "q",
        n=-5,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(should_not_be_called),
    )
    assert out == []


# ---------------------------------------------------------------------------
# hyde_embed — the averaging layer
# ---------------------------------------------------------------------------

async def test_hyde_embed_averages_excerpt_and_query():
    """One excerpt + raw query → unit-normalized component-wise mean.

    Per review §3.7 the averaged vector is renormalized to unit norm
    before return — Qdrant's cosine distance assumes unit-norm queries
    (TEI returns unit vectors), so the sum-of-unit-vectors mean must be
    rescaled or distance scoring is silently miscalibrated.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("the excerpt text")

    embedder = _FixedEmbedder({
        "the excerpt text": [1.0, 2.0, 3.0, 4.0],
        "the raw query": [0.0, 0.0, 5.0, 8.0],
    })

    avg = await hyde_embed(
        "the raw query",
        embedder,
        n=1,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=True,
        transport=_make_transport(handler),
    )
    # mean = [0.5, 1.0, 4.0, 6.0]; normalized → unit vector with same direction.
    expected = _normalize([0.5, 1.0, 4.0, 6.0])
    assert avg == pytest.approx(expected)
    assert _norm(avg) == pytest.approx(1.0)
    # Single batch embed call.
    assert len(embedder.calls) == 1
    # Order is [excerpts..., raw_query]
    assert embedder.calls[0] == ["the excerpt text", "the raw query"]


async def test_hyde_embed_without_raw_query():
    """include_raw_query=False → average only the excerpts (unit-normalized)."""

    call = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call["n"] += 1
        return _chat_response(f"excerpt-{call['n']}")

    embedder = _FixedEmbedder({
        "excerpt-1": [1.0, 1.0, 1.0],
        "excerpt-2": [3.0, 3.0, 3.0],
    })

    avg = await hyde_embed(
        "the raw query",
        embedder,
        n=2,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=False,
        transport=_make_transport(handler),
    )
    # Mean of [1,1,1] and [3,3,3] → [2,2,2]; normalize → 1/sqrt(3) each.
    expected = _normalize([2.0, 2.0, 2.0])
    assert avg == pytest.approx(expected)
    assert _norm(avg) == pytest.approx(1.0)
    assert embedder.calls[0] == ["excerpt-1", "excerpt-2"]


async def test_hyde_embed_returns_none_when_all_generations_fail():
    """All excerpts fail → return None (caller will embed raw query)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("", status=500)

    embedder = _FixedEmbedder({}, default=[9.0, 9.0])

    out = await hyde_embed(
        "the raw query",
        embedder,
        n=3,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        transport=_make_transport(handler),
    )
    assert out is None
    # No embedder call at all — we short-circuited.
    assert embedder.calls == []


async def test_hyde_embed_partial_success_still_averages():
    """One of two succeeds → we still return a (smaller) average."""

    call = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call["n"] += 1
        if call["n"] == 1:
            return _chat_response("good excerpt")
        return _chat_response("", status=502)

    embedder = _FixedEmbedder({
        "good excerpt": [2.0, 4.0],
        "q": [0.0, 0.0],
    })

    avg = await hyde_embed(
        "q",
        embedder,
        n=2,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=True,
        transport=_make_transport(handler),
    )
    # 1 excerpt + raw query → mean([2,4], [0,0]) = [1.0, 2.0]; unit-norm.
    expected = _normalize([1.0, 2.0])
    assert avg == pytest.approx(expected)
    assert _norm(avg) == pytest.approx(1.0)


async def test_hyde_embed_n_equal_three_averages_all():
    """N=3 all succeed + raw query → mean across 4 vectors."""

    call = {"n": 0}
    texts = ["excerpt-alpha", "excerpt-beta", "excerpt-gamma"]

    def handler(req: httpx.Request) -> httpx.Response:
        idx = call["n"] % 3
        call["n"] += 1
        return _chat_response(texts[idx])

    embedder = _FixedEmbedder({
        "excerpt-alpha": [1.0, 0.0],
        "excerpt-beta": [0.0, 1.0],
        "excerpt-gamma": [1.0, 1.0],
        "q": [0.0, 0.0],
    })

    avg = await hyde_embed(
        "q",
        embedder,
        n=3,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=True,
        transport=_make_transport(handler),
    )
    # Set comparison to tolerate handler-call ordering in asyncio.gather —
    # we assert the SUM of the 4 vectors was [2, 2] → mean [0.5, 0.5];
    # post §3.7 normalization → [1/sqrt(2), 1/sqrt(2)] (unit norm).
    assert avg is not None
    assert len(avg) == 2
    assert avg[0] == pytest.approx(1 / math.sqrt(2))
    assert avg[1] == pytest.approx(1 / math.sqrt(2))
    assert _norm(avg) == pytest.approx(1.0)


async def test_hyde_embed_unit_inputs_produce_unit_output():
    """Sanity check (§3.7): mean of unit vectors is renormalized to unit.

    TEI returns unit-norm embeddings. The pre-fix average produced a
    sub-unit vector whose direction was correct but whose magnitude
    biased Qdrant's cosine-distance scoring. Post-fix, every output is
    unit-norm regardless of input magnitudes.
    """
    # Two orthogonal unit vectors → mean magnitude is 1/sqrt(2),
    # post-normalization it should be exactly 1.
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("excerpt-a")

    embedder = _FixedEmbedder({"excerpt-a": a, "q": b})
    avg = await hyde_embed(
        "q",
        embedder,
        n=1,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=True,
        transport=_make_transport(handler),
    )
    assert avg is not None
    assert _norm(avg) == pytest.approx(1.0)


async def test_hyde_embed_zero_average_falls_back_to_input():
    """Edge case (§3.7): if the averaged vector is exactly zero (e.g. two
    inputs that cancel), normalization would divide by zero. The fix
    must guard against this without crashing — returning the unscaled
    average is acceptable since downstream cosine treats it as undefined
    direction; what is NOT acceptable is a divide-by-zero exception."""

    def handler(req: httpx.Request) -> httpx.Response:
        return _chat_response("opposite excerpt")

    embedder = _FixedEmbedder({
        "opposite excerpt": [1.0, 0.0, 0.0],
        "q": [-1.0, 0.0, 0.0],
    })

    avg = await hyde_embed(
        "q",
        embedder,
        n=1,
        chat_url=CHAT_URL,
        chat_model=CHAT_MODEL,
        include_raw_query=True,
        transport=_make_transport(handler),
    )
    # Doesn't crash; result is a finite vector (or all zeros).
    assert avg is not None
    assert all(math.isfinite(x) for x in avg)


# ---------------------------------------------------------------------------
# is_enabled — flag helper
# ---------------------------------------------------------------------------

def test_is_enabled_default_false():
    """With no flag set anywhere, ``is_enabled()`` returns False."""
    # flags.get falls through to os.environ; pytest's monkeypatch fixture
    # unset would be needed for process-level guarantee — use the overlay
    # to force a known state.
    with flags.with_overrides({"RAG_HYDE": "0"}):
        assert is_enabled() is False


def test_is_enabled_true_when_overlay_set():
    """Overlay-level RAG_HYDE=1 → True."""
    with flags.with_overrides({"RAG_HYDE": "1"}):
        assert is_enabled() is True


def test_is_enabled_strict_on_one():
    """Only exact ``'1'`` is truthy — ``'true'`` is not."""
    with flags.with_overrides({"RAG_HYDE": "true"}):
        assert is_enabled() is False
    with flags.with_overrides({"RAG_HYDE": "yes"}):
        assert is_enabled() is False
    with flags.with_overrides({"RAG_HYDE": ""}):
        assert is_enabled() is False
