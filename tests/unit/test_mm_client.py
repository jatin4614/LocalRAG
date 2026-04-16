import pytest
import httpx
from model_manager.client import ModelClient, ModelClientError


@pytest.mark.asyncio
async def test_wake_success_returns_true():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/wake_up"
        return httpx.Response(200, json={"state": "awake"})

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.wake_up() is True
    await client.aclose()


@pytest.mark.asyncio
async def test_sleep_success_returns_true():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sleep"
        return httpx.Response(200, json={"state": "asleep"})

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.sleep_model() is True
    await client.aclose()


@pytest.mark.asyncio
async def test_health_awake_when_reachable():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.health() == "awake"
    await client.aclose()


@pytest.mark.asyncio
async def test_health_unknown_on_connection_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.health() == "unknown"
    await client.aclose()


@pytest.mark.asyncio
async def test_wake_raises_on_5xx():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    with pytest.raises(ModelClientError):
        await client.wake_up()
    await client.aclose()
