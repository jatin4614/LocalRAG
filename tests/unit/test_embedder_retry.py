"""Unit tests for ``ext.services.embedder.TEIEmbedder`` redundancy layer.

Background — failure mode being fixed:
    GPU 1 (24 GB shared by TEI + reranker + colbert + fastembed + vllm-qu)
    runs at ~95% steady-state. When the celery ingest worker fans a batch
    of chunks at TEI, TEI's per-forward activation can push the GPU over
    and the forward returns 424 (`DriverError(CUDA_ERROR_OUT_OF_MEMORY)`).

    Pre-fix, ``embed()`` raised the 424 and the celery task failed with no
    retry. The doc landed at ``ingest_status='failed'`` and the operator
    had to soft-delete + re-upload manually.

    Fix: retry-with-halving inside ``embed``. On a retryable HTTP error
    (424 / 429 / 5xx / network), retry up to ``RAG_EMBED_MAX_RETRIES``
    times at the same batch size; if still failing AND batch size > 1,
    halve the batch and recurse. Halving stops at batch=1 (real per-chunk
    failure surfaces).

These tests use ``httpx.MockTransport`` so they NEVER hit a real TEI.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from ext.services.embedder import TEIEmbedder


@pytest.fixture(autouse=True)
def _disable_breaker_and_short_backoff(monkeypatch):
    """Default test environment.

    * Disable the circuit breaker — tests focus on retry-with-halving.
      Breaker behaviour has its own coverage in
      ``tests/unit/test_tei_circuit_breaker.py``.
    * Force fast retry sleeps so a 3-attempt × 4-batch-level recursion
      doesn't take 30s of real time. Production defaults are 0.5/1.0/2.0;
      tests use ~1ms between attempts via the patched ``asyncio.sleep``.
    """
    monkeypatch.setenv("RAG_CB_TEI_ENABLED", "0")
    monkeypatch.setenv("RAG_CIRCUIT_BREAKER_ENABLED", "0")
    yield


@pytest.fixture
def fast_sleep(monkeypatch):
    """Patch ``asyncio.sleep`` so backoff waits collapse to ~0s.

    The embedder module uses ``await asyncio.sleep(...)`` for backoff;
    patching the symbol on ``asyncio`` itself catches both
    ``import asyncio; asyncio.sleep`` and ``from asyncio import sleep``
    forms (since the test relies on production code re-using the global
    binding).
    """
    sleeps: list[float] = []

    async def _no_sleep(d):  # noqa: D401 — drop-in replacement
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    return sleeps


def _make_handler(responses: list[httpx.Response]):
    """Build a MockTransport handler that returns each Response in order.

    On exhaustion raises so we don't accidentally mask "too few mocks
    configured" with infinite repeats.
    """
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = state["i"]
        if i >= len(responses):
            raise AssertionError(
                f"MockTransport exhausted after {i} responses; "
                "test forgot to enqueue a response or the embed loop "
                "made more calls than expected"
            )
        state["i"] += 1
        return responses[i]

    return handler, state


# --------------------------------------------------------------------------
# Test 1: happy path unchanged
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_happy_path_no_retry(fast_sleep):
    """A mocked TEI returning 200 on the first call → embed returns the
    right number of vectors and does NOT trigger the retry layer.
    """
    handler, state = _make_handler([
        httpx.Response(200, json=[[0.1, 0.2, 0.3]]),
    ])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    out = await e.embed(["hello"])

    assert out == [[0.1, 0.2, 0.3]]
    assert state["i"] == 1
    assert fast_sleep == []  # no backoff sleeps
    await e.aclose()


# --------------------------------------------------------------------------
# Test 2: single 424 then success — retry counter bumps with outcome=recovered
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_424_then_success_bumps_recovered_counter(
    monkeypatch, fast_sleep
):
    """Mock returns 424 once, then 200. embed() succeeds; the
    ``embedder_retry_total{outcome="recovered", reason="424"}`` counter
    is bumped at least once.
    """
    handler, state = _make_handler([
        httpx.Response(424, json={"error": "OOM"}),
        httpx.Response(200, json=[[0.4, 0.5]]),
    ])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    # Snapshot the metric counter before the call.
    from ext.services import metrics as M

    before = _counter_value(
        M.embedder_retry_total, outcome="recovered", reason="424"
    )

    out = await e.embed(["hello"])

    assert out == [[0.4, 0.5]]
    assert state["i"] == 2  # 424, then 200
    after = _counter_value(
        M.embedder_retry_total, outcome="recovered", reason="424"
    )
    assert after - before == 1
    await e.aclose()


# --------------------------------------------------------------------------
# Test 3: persistent 424 at batch=8 triggers halving to batch=4 and recovers
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_persistent_424_triggers_halving(monkeypatch, fast_sleep):
    """Mock returns 424 N times in a row at batch 8; assert that halving
    fires (batch shrinks to 4) and eventually succeeds. The halving
    counter is bumped with the right size class.
    """
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "3")
    monkeypatch.setenv("RAG_TEI_MAX_BATCH", "8")

    # 3 retries at batch 8, then halve → 2 successful calls at batch 4.
    handler, state = _make_handler([
        httpx.Response(424, json={"error": "OOM"}),  # batch 8 attempt 1
        httpx.Response(424, json={"error": "OOM"}),  # batch 8 attempt 2
        httpx.Response(424, json={"error": "OOM"}),  # batch 8 attempt 3
        httpx.Response(200, json=[[0.0]] * 4),       # batch 4 (first half)
        httpx.Response(200, json=[[0.0]] * 4),       # batch 4 (second half)
    ])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    from ext.services import metrics as M

    halving_before = _counter_value(
        M.embedder_halving_total, batch_size_class="2-4"
    )

    out = await e.embed(["t"] * 8)

    assert len(out) == 8
    assert state["i"] == 5
    halving_after = _counter_value(
        M.embedder_halving_total, batch_size_class="2-4"
    )
    assert halving_after - halving_before == 1  # one halving 8 → 4
    await e.aclose()


# --------------------------------------------------------------------------
# Test 4: halving recurses all the way to batch=1, then exhausts
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_halving_recurses_to_batch_1_then_raises(monkeypatch, fast_sleep):
    """Mock returns 424 always. Assert batch shrinks 8 → 4 → 2 → 1, each
    at the configured retry budget, then raises with a clear error.
    ``embedder_retry_total{outcome="exhausted"}`` is bumped at least once.
    """
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "2")  # smaller for speed
    monkeypatch.setenv("RAG_TEI_MAX_BATCH", "8")

    # Always 424. Test must not depend on exact call count beyond the
    # invariant that all calls saw 424.
    def handler(request):
        return httpx.Response(424, json={"error": "OOM"})

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    from ext.services import metrics as M

    exhausted_before = _counter_value(
        M.embedder_retry_total, outcome="exhausted", reason="424"
    )
    halving_8to4 = _counter_value(M.embedder_halving_total, batch_size_class="2-4")
    halving_4to2 = _counter_value(M.embedder_halving_total, batch_size_class="2-4")
    halving_to1 = _counter_value(M.embedder_halving_total, batch_size_class="1")

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await e.embed(["t"] * 8)

    # Error message identifies a 424 (the original retryable cause).
    assert excinfo.value.response.status_code == 424

    # Halving fires at every step down: 8→4 (size class 2-4), 4→2 (size
    # class 2-4), 2→1 (size class 1).
    halving_2_4_after = _counter_value(
        M.embedder_halving_total, batch_size_class="2-4"
    )
    halving_1_after = _counter_value(M.embedder_halving_total, batch_size_class="1")
    assert halving_2_4_after - halving_8to4 >= 2  # 8→4 and 4→2
    assert halving_1_after - halving_to1 >= 1     # 2→1

    # Exhausted counter bumped (at least once at the batch=1 floor).
    exhausted_after = _counter_value(
        M.embedder_retry_total, outcome="exhausted", reason="424"
    )
    assert exhausted_after - exhausted_before >= 1
    await e.aclose()


# --------------------------------------------------------------------------
# Test 5: order preservation across halving — alignment is load-bearing
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_order_preserved_across_halving(monkeypatch, fast_sleep):
    """Input 8 distinct texts, mock 424 first then 200 on each half.
    Verify returned vectors are in the SAME ORDER as input texts.
    """
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "1")
    monkeypatch.setenv("RAG_TEI_MAX_BATCH", "8")

    # Each half-batch produces its own distinguishable vector pattern so
    # we can assert positional alignment.
    first_half = [[1.0, float(i)] for i in range(4)]
    second_half = [[2.0, float(i)] for i in range(4)]

    handler, state = _make_handler([
        httpx.Response(424, json={"error": "OOM"}),  # batch 8 fails
        httpx.Response(200, json=first_half),         # halves[0]
        httpx.Response(200, json=second_half),        # halves[1]
    ])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    inputs = ["a", "b", "c", "d", "e", "f", "g", "h"]
    out = await e.embed(inputs)

    assert len(out) == 8
    # First half markers (vec[0]==1.0) cover positions 0..3,
    # second half markers (vec[0]==2.0) cover positions 4..7.
    assert all(v[0] == 1.0 for v in out[:4]), out
    assert all(v[0] == 2.0 for v in out[4:]), out
    # Index within each half preserved (0,1,2,3 not 3,2,1,0).
    assert [v[1] for v in out[:4]] == [0.0, 1.0, 2.0, 3.0]
    assert [v[1] for v in out[4:]] == [0.0, 1.0, 2.0, 3.0]
    await e.aclose()


# --------------------------------------------------------------------------
# Test 6: non-retryable 4xx surfaces immediately
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_non_retryable_422_surfaces_immediately(monkeypatch, fast_sleep):
    """A 422 (real input bug) returns immediately as exception. No retry,
    no halving — surface the operator-actionable error.
    """
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "3")

    handler, state = _make_handler([
        httpx.Response(422, json={"error": "input too long"}),
    ])
    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await e.embed(["bad input"])

    assert excinfo.value.response.status_code == 422
    assert state["i"] == 1  # exactly one call, no retry
    await e.aclose()


# --------------------------------------------------------------------------
# Test 7: httpx.ReadTimeout is retryable
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_read_timeout_then_success(monkeypatch, fast_sleep):
    """Same shape as the 424-then-success test but for ReadTimeout."""
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "3")

    state = {"i": 0}

    def handler(request):
        state["i"] += 1
        if state["i"] == 1:
            raise httpx.ReadTimeout("tei stalled", request=request)
        return httpx.Response(200, json=[[9.9]])

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    out = await e.embed(["x"])
    assert out == [[9.9]]
    assert state["i"] == 2

    from ext.services import metrics as M

    # Should have bumped recovered/network at least once.
    recovered = _counter_value(
        M.embedder_retry_total, outcome="recovered", reason="network"
    )
    assert recovered >= 1
    await e.aclose()


# --------------------------------------------------------------------------
# Test 8: empty input returns empty without HTTP call
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_input_returns_empty(fast_sleep):
    """``embed([])`` returns ``[]`` without making a request."""
    state = {"i": 0}

    def handler(request):
        state["i"] += 1
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    out = await e.embed([])
    assert out == []
    assert state["i"] == 0  # no HTTP call
    await e.aclose()


# --------------------------------------------------------------------------
# Test 9: batch=1 is the recursion floor — never tries to halve below 1
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batch_one_floor_does_not_recurse(monkeypatch, fast_sleep):
    """Single text input + persistent 424 must terminate at the configured
    retry budget, NOT loop forever or attempt to divide a 1-element list.
    """
    monkeypatch.setenv("RAG_EMBED_MAX_RETRIES", "3")

    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        return httpx.Response(424, json={"error": "OOM"})

    transport = httpx.MockTransport(handler)
    e = TEIEmbedder(base_url="http://tei", transport=transport)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await e.embed(["only one text"])

    assert excinfo.value.response.status_code == 424
    # Exactly RAG_EMBED_MAX_RETRIES attempts, no halving recursion below 1.
    assert call_count["n"] == 3
    await e.aclose()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _counter_value(metric, **labels) -> float:
    """Read the current value of a labelled prometheus counter.

    Returns 0.0 if prometheus_client is missing (the noop sentinel has
    no value backing) — tests that exercise counter behaviour are
    skipped at the assertion site by checking before/after deltas.
    """
    try:
        return float(metric.labels(**labels)._value.get())
    except (AttributeError, KeyError):
        return 0.0
