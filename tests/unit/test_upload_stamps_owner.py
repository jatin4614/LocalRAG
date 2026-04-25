"""Unit tests for P2.2 ``owner_user_id`` stamping in ``ext.routers.upload``.

Both upload paths must stamp ``owner_user_id`` into ``payload_base`` so that
every chunk upserted into Qdrant carries the originating user's id.

  * KB path (``/api/kb/{kb_id}/subtag/{subtag_id}/upload``) — stamps the
    uploading admin's id. Acts as an audit trail; retrieval does NOT filter
    KB collections by owner (KBs are shared).
  * Private-chat path (``/api/chats/{chat_id}/private_docs/upload``) —
    stamps the chat owner's id. Retrieval DOES filter by owner for the
    chat-scoped namespace (per-user isolation).

Strategy: stub ``ingest_bytes`` at the module level and capture the
``payload_base`` kwarg passed in. No DB, no vector store, no Qdrant.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.routers import upload as upload_mod


def _reset_module() -> None:
    """Reset the module-level singletons so each test starts clean."""
    upload_mod._SM = None
    upload_mod._VS = None
    upload_mod._EMB = None


async def _fake_flush(self):  # noqa: ARG001
    # Simulate SQLAlchemy assigning a primary key after flush so the doc's
    # id is not None when ingest_bytes is called.
    pass


class _FakeResult:
    """Wrap a value so both ``.scalar_one_or_none()`` and ``.first()`` work.

    ``first()`` returns the raw value verbatim so callers can pass a tuple
    like ``("rag-cfg-json",)`` and subscript ``row[0]`` without surprises.
    Single-value (non-tuple) callers must continue to use
    ``scalar_one_or_none()`` — that's what upload.py does for the subtag
    lookup, and what private-chat upload does for the chat-row check via
    ``first()`` (already passes a tuple).
    """

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def first(self):
        return self._value


class _FakeSession:
    """Minimal async-session stand-in. ``session.add`` stamps a numeric id
    onto the KBDocument so downstream code sees a non-None ``doc.id``.

    ``execute_return`` may be a single value (one execute call) or a list
    of values (multiple execute calls — returned in FIFO order). The
    upload_kb_doc handler calls ``execute`` twice — first for the subtag
    lookup (consumed via ``scalar_one_or_none``), then for the KB
    rag_config row (consumed via ``first()``, must be subscriptable).
    Tests that only exercise one execute call may keep passing a single
    value for back-compat.
    """

    def __init__(self, *, execute_return=None):
        # Coerce single value into a one-element queue for uniform
        # consumption inside ``execute``.
        if isinstance(execute_return, list):
            self._execute_queue = list(execute_return)
        else:
            self._execute_queue = [execute_return]
        self.added: list = []

    def add(self, obj):
        # Simulate DB-assigned PK.
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = 123
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def execute(self, *a, **kw):  # noqa: ARG002
        # FIFO: each execute call consumes one return value. After the
        # queue is drained, repeat the last value indefinitely so
        # accidental extra execute calls don't crash with IndexError.
        if len(self._execute_queue) > 1:
            value = self._execute_queue.pop(0)
        else:
            value = self._execute_queue[0]
        return _FakeResult(value)


# ---------- KB upload path ---------------------------------------------------


@pytest.mark.asyncio
async def test_kb_upload_stamps_owner_user_id(monkeypatch) -> None:
    """KB admin upload path: payload_base must include ``owner_user_id`` =
    the uploading admin's id. Mirrors the private-chat invariant."""
    _reset_module()
    upload_mod.RAG_SYNC_INGEST = True
    upload_mod._VS = MagicMock()
    upload_mod._VS.ensure_collection = AsyncMock()
    upload_mod._EMB = object()

    # Capture what ingest_bytes is called with.
    captured: dict = {}

    async def fake_ingest_bytes(**kwargs):
        captured.update(kwargs)
        return 3  # n chunks

    monkeypatch.setattr(upload_mod, "ingest_bytes", fake_ingest_bytes)

    # Stub kb_service.get_kb → truthy KB object.
    async def fake_get_kb(session, *, kb_id):  # noqa: ARG001
        return SimpleNamespace(id=kb_id)

    monkeypatch.setattr(upload_mod.kb_service, "get_kb", fake_get_kb)

    # Two execute calls happen inside upload_kb_doc:
    #   (1) subtag lookup → scalar_one_or_none → SimpleNamespace
    #   (2) KB.rag_config lookup → first() → tuple-like row
    # Pass a list so the FIFO _FakeSession.execute hands the right thing
    # to each. (None,) for the rag_config row signals "no per-KB config",
    # which makes resolve_chunk_params fall back to env defaults.
    subtag = SimpleNamespace(id=100, kb_id=10)
    session = _FakeSession(execute_return=[subtag, (None,)])

    # Build an UploadFile stand-in. ``_read_bounded`` does ``await file.read(...)``
    # in a loop; a minimal async generator stand-in is enough.
    class _FakeFile:
        def __init__(self, data: bytes, filename: str, content_type: str) -> None:
            self._data = data
            self._off = 0
            self.filename = filename
            self.content_type = content_type

        async def read(self, n: int = -1) -> bytes:
            if self._off >= len(self._data):
                return b""
            chunk = self._data[self._off:self._off + n] if n and n > 0 else self._data[self._off:]
            self._off += len(chunk)
            return chunk

    file = _FakeFile(b"hello world payload base test", "a.txt", "text/plain")
    user = SimpleNamespace(id="admin-uuid-aaa", role="admin")

    result = await upload_mod.upload_kb_doc(
        kb_id=10,
        subtag_id=100,
        file=file,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
    )
    assert result.status == "done"
    assert result.chunks == 3
    pb = captured["payload_base"]
    assert pb["kb_id"] == 10
    assert pb["subtag_id"] == 100
    assert pb["doc_id"] == 123  # assigned by _FakeSession.add
    assert pb["owner_user_id"] == "admin-uuid-aaa"


@pytest.mark.asyncio
async def test_kb_upload_stamps_owner_user_id_numeric(monkeypatch) -> None:
    """When running in stub AUTH_MODE the user id is a numeric string
    (e.g. ``"9"``) — payload_base must carry it verbatim."""
    _reset_module()
    upload_mod.RAG_SYNC_INGEST = True
    upload_mod._VS = MagicMock()
    upload_mod._VS.ensure_collection = AsyncMock()
    upload_mod._EMB = object()

    captured: dict = {}

    async def fake_ingest_bytes(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(upload_mod, "ingest_bytes", fake_ingest_bytes)

    async def fake_get_kb(session, *, kb_id):  # noqa: ARG001
        return SimpleNamespace(id=kb_id)

    monkeypatch.setattr(upload_mod.kb_service, "get_kb", fake_get_kb)

    # See sibling test for why execute_return is a 2-element list:
    # subtag for the first execute, (None,) tuple for the rag_config row.
    subtag = SimpleNamespace(id=100, kb_id=10)
    session = _FakeSession(execute_return=[subtag, (None,)])

    class _FakeFile:
        def __init__(self):
            self._d = b"xyz hello"
            self._off = 0
            self.filename = "a.txt"
            self.content_type = "text/plain"

        async def read(self, n=-1):
            if self._off >= len(self._d):
                return b""
            out = self._d[self._off:self._off + n] if n and n > 0 else self._d[self._off:]
            self._off += len(out)
            return out

    await upload_mod.upload_kb_doc(
        kb_id=10,
        subtag_id=100,
        file=_FakeFile(),  # type: ignore[arg-type]
        user=SimpleNamespace(id="9", role="admin"),  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
    )
    assert captured["payload_base"]["owner_user_id"] == "9"


# ---------- Private-chat upload path ----------------------------------------


@pytest.mark.asyncio
async def test_private_chat_upload_stamps_owner_user_id(monkeypatch) -> None:
    """Private-chat upload path: payload_base must include
    ``owner_user_id`` (the chat owner). Pre-existing wiring — protect via
    an explicit unit test so a future refactor can't silently drop the field.
    """
    _reset_module()
    upload_mod.RAG_SYNC_INGEST = True
    upload_mod._VS = MagicMock()
    upload_mod._VS.ensure_collection = AsyncMock()
    upload_mod._EMB = object()

    captured: dict = {}

    async def fake_ingest_bytes(**kwargs):
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(upload_mod, "ingest_bytes", fake_ingest_bytes)

    # For the private-doc route, session.execute returns the chat row:
    # (user_id,) where the caller-supplied user.id must match.
    session = _FakeSession(execute_return=("user-uuid-bbb",))

    class _FakeFile:
        def __init__(self):
            self._d = b"private note with some payload for chunking"
            self._off = 0
            self.filename = "note.txt"
            self.content_type = "text/plain"

        async def read(self, n=-1):
            if self._off >= len(self._d):
                return b""
            out = self._d[self._off:self._off + n] if n and n > 0 else self._d[self._off:]
            self._off += len(out)
            return out

    result = await upload_mod.upload_private_doc(
        chat_id="chat-xyz",
        file=_FakeFile(),  # type: ignore[arg-type]
        user=SimpleNamespace(id="user-uuid-bbb", role="user"),  # type: ignore[arg-type]
        session=session,  # type: ignore[arg-type]
    )
    assert result.status == "done"
    assert result.chunks == 2
    pb = captured["payload_base"]
    assert pb["chat_id"] == "chat-xyz"
    assert pb["owner_user_id"] == "user-uuid-bbb"
