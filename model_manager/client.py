"""HTTP client for a single backend model server (vllm or whisper wrapper)."""
from __future__ import annotations

from typing import Literal, Optional

import httpx
from httpx import AsyncBaseTransport


class ModelClientError(RuntimeError):
    """Raised on 4xx/5xx or transport failure when the caller expected success."""


HealthState = Literal["awake", "asleep", "unknown"]


class ModelClient:
    def __init__(
        self, *, base_url: str, health_path: str,
        timeout: float = 10.0, transport: Optional[AsyncBaseTransport] = None,
    ) -> None:
        self._base_url = base_url
        self._health_path = health_path
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> HealthState:
        """Light probe: 2xx → awake, connection error → unknown."""
        try:
            r = await self._client.get(self._health_path)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            return "unknown"
        if r.status_code < 300:
            return "awake"
        return "unknown"

    async def wake_up(self) -> bool:
        try:
            r = await self._client.post("/wake_up")
        except httpx.HTTPError as e:
            raise ModelClientError(f"wake_up transport failure: {e}") from e
        if r.status_code >= 500:
            raise ModelClientError(f"wake_up returned {r.status_code}")
        if r.status_code >= 400:
            raise ModelClientError(f"wake_up 4xx: {r.status_code} — backend rejected request")
        return True

    async def sleep_model(self) -> bool:
        try:
            r = await self._client.post("/sleep")
        except httpx.HTTPError as e:
            raise ModelClientError(f"sleep transport failure: {e}") from e
        if r.status_code >= 500:
            raise ModelClientError(f"sleep returned {r.status_code}")
        if r.status_code >= 400:
            raise ModelClientError(f"sleep 4xx: {r.status_code}")
        return True
