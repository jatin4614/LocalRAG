import httpx
import pytest

from ext.services.retry_policy import with_transient_retry, is_transient


def test_is_transient_timeout():
    assert is_transient(httpx.TimeoutException("t"))
    assert is_transient(httpx.ConnectError("c"))
    assert is_transient(httpx.ReadError("r"))


def test_is_transient_5xx():
    resp = httpx.Response(502, request=httpx.Request("GET", "http://x"))
    err = httpx.HTTPStatusError("5xx", request=resp.request, response=resp)
    assert is_transient(err)


def test_is_transient_4xx_is_not_transient():
    resp = httpx.Response(400, request=httpx.Request("GET", "http://x"))
    err = httpx.HTTPStatusError("400", request=resp.request, response=resp)
    assert not is_transient(err)


def test_is_transient_generic_exception_is_not_transient():
    assert not is_transient(ValueError("bad input"))


@pytest.mark.asyncio
async def test_retry_wrapper_retries_transient_then_succeeds():
    calls = {"n": 0}

    @with_transient_retry(attempts=3, base_sec=0.01)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.TimeoutException("timeout")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_wrapper_does_not_retry_non_transient():
    calls = {"n": 0}

    @with_transient_retry(attempts=3, base_sec=0.01)
    async def bad():
        calls["n"] += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        await bad()
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_wrapper_feature_flag_off_passes_through(monkeypatch):
    monkeypatch.setenv("RAG_TENACITY_RETRY", "0")
    calls = {"n": 0}

    @with_transient_retry(attempts=3, base_sec=0.01)
    async def flaky():
        calls["n"] += 1
        raise httpx.TimeoutException("t")

    with pytest.raises(httpx.TimeoutException):
        await flaky()
    assert calls["n"] == 1
