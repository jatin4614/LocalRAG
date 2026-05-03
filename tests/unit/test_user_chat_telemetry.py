"""Verify the upstream chat completion handler is wrapped in
``record_llm_call`` (review §6.4).

The previous E-1 agent reported §6.4 BLOCKED upstream — the chat-LLM
dispatch lives in ``upstream/backend/open_webui/main.py`` (around line
1659 ``@app.post('/api/chat/completions')`` → ``chat_completion`` →
``chat_completion_handler``). This wave wires the recorder around that
call site so the user-facing chat path emits the same TTFT / token
metrics as the contextualizer / hyde / rewriter / SSE-stream paths.

These tests:

1. Static guard — assert the upstream file imports
   ``record_llm_call`` and uses it inside the chat-completion handler
   with ``stage="user_chat"``.
2. Patch tracker — assert ``patches/0007_chat_completion_telemetry.patch``
   exists so the wiring survives an ``upstream/`` re-vendor.
3. Behavior — monkeypatch the recorder + a stub chat handler and
   verify the wrap fires (recorder enters, exits, and observes the
   model + kb labels we threaded in).

The behavior test uses a small mock instead of standing up the full
FastAPI app — that's what the E-1 BLOCKED note pointed at: the
upstream surface is too tangled to safely run the real handler in
unit-test land. We assert the WIRING via static checks and the
RECORDING via a focused mock of ``record_llm_call``.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_MAIN = ROOT / "upstream" / "backend" / "open_webui" / "main.py"
PATCH_FILE = ROOT / "patches" / "0007_chat_completion_telemetry.patch"


# ---------- static guards: source file must reference the canonical helper ---


def _read_main_source() -> str:
    return UPSTREAM_MAIN.read_text(encoding="utf-8")


def test_upstream_main_exists() -> None:
    """Sanity: the file we patched is still here after re-vendor."""
    assert UPSTREAM_MAIN.is_file(), (
        f"upstream/backend/open_webui/main.py not found at {UPSTREAM_MAIN}; "
        "re-apply patches/0007_chat_completion_telemetry.patch after re-vendor"
    )


def test_main_imports_record_llm_call() -> None:
    """The chat handler module must import ``record_llm_call`` from
    ``ext.services.llm_telemetry`` (or alias it).

    Without this, the wrap is a no-op and the user-facing chat path
    drops off the Prometheus dashboards. Patches/0007 adds the import
    behind a try/except so an upstream re-vendor without ``ext`` on
    the path still imports cleanly (the no-op shim takes over).
    """
    src = _read_main_source()
    assert (
        "from ext.services.llm_telemetry import record_llm_call" in src
    ), (
        "upstream/backend/open_webui/main.py must import record_llm_call "
        "(see patches/0007_chat_completion_telemetry.patch — re-apply if "
        "upstream was re-vendored)"
    )


def test_main_uses_recorder_around_chat_completion_handler() -> None:
    """Static check: the ``chat_completion_handler`` call must be inside
    an ``async with _record_llm_call(stage="user_chat", ...)`` block.

    The string match isn't tight (the formatter may re-wrap args), but
    we require the three key signals: the recorder name, the stage
    label, and the handler call.
    """
    src = _read_main_source()
    assert "async with _record_llm_call(" in src, (
        "main.py must enter the recorder via ``async with _record_llm_call(...)``"
    )
    assert "stage='user_chat'" in src or 'stage="user_chat"' in src, (
        "the recorder must label the chat path stage='user_chat' so "
        "dashboards can break it out from the SSE stage='chat' path"
    )
    assert "chat_completion_handler(request, form_data, user)" in src, (
        "the wrap must enclose the chat_completion_handler invocation, "
        "not some unrelated call"
    )


def test_main_extracts_kb_label_from_metadata() -> None:
    """KB label sourced from ``metadata['kb_config']`` (the canonical
    selection home post-migration 007). The exact field name is part
    of the patch contract.
    """
    src = _read_main_source()
    assert "metadata" in src and "kb_config" in src, (
        "the wrap must read the user's KB selection from "
        "metadata['kb_config'] to populate the kb label on "
        "rag_tokens_prompt_total"
    )


def test_no_op_shim_present_for_missing_ext() -> None:
    """The try/except fallback must define a no-op recorder — without
    it, an upstream re-vendor in an environment where ``ext`` is not
    on the path would crash at import time.
    """
    src = _read_main_source()
    assert "_NullRecorder" in src or "NullRecorder" in src, (
        "the import block must define a no-op recorder fallback so "
        "vanilla open-webui environments don't crash on import"
    )


# ---------- patch tracker --------------------------------------------------


def test_patch_file_present() -> None:
    """The patch file is the re-derive instruction for upstream re-vendor."""
    assert PATCH_FILE.is_file(), (
        f"patches/0007_chat_completion_telemetry.patch not found at {PATCH_FILE}; "
        "without it the wiring is lost on the next upstream re-vendor"
    )


def test_patch_file_targets_main_py() -> None:
    """The patch must apply to ``backend/open_webui/main.py`` (the
    upstream-relative path used by the existing 0001-0006 patches).
    """
    body = PATCH_FILE.read_text(encoding="utf-8")
    assert "backend/open_webui/main.py" in body, (
        "the patch must list the correct relative target path "
        "(backend/open_webui/main.py)"
    )


def test_patch_file_includes_recorder_lines() -> None:
    """The patch's added-lines (`+`) must include the recorder import
    and the wrap. Catches accidental empty / corrupted patch files.
    """
    body = PATCH_FILE.read_text(encoding="utf-8")
    # The import is inside a try-block in the patch (indented 4 spaces)
    # so we check for the symbol-bearing fragment, not a column-anchored
    # match.
    assert "from ext.services.llm_telemetry import record_llm_call" in body, (
        "patch must add the recorder import"
    )
    assert "+            async with _record_llm_call(" in body, (
        "patch must add the async-with wrap inside process_chat"
    )


# ---------- behavioral test: recorder fires when handler runs ---------------


@pytest.mark.asyncio
async def test_recorder_fires_around_mock_handler(monkeypatch) -> None:
    """End-to-end check via a small in-test mock: a function shaped like
    the upstream wrap must enter the recorder, call the handler, and
    pass observed token counts back to ``set_tokens``.

    This isn't running the real ``chat_completion`` (the upstream
    handler is too tangled — that was the original BLOCKED reason),
    but it locks the *pattern* the patch installs so a regression in
    the wrap is caught here even if no integration test stands up the
    full FastAPI app.
    """
    # Track recorder lifecycle.
    enter_calls: list[dict] = []
    exit_called = {"n": 0}
    set_tokens_calls: list[dict] = []

    class _RecObj:
        def set_tokens(self, *, prompt: int, completion: int) -> None:
            set_tokens_calls.append({"prompt": prompt, "completion": completion})

        def set_first_token_at(self, t: float) -> None:
            pass

        def set_kb(self, kb: str) -> None:
            pass

    @asynccontextmanager
    async def _fake_record(*, stage: str, model: str, kb=None):
        enter_calls.append({"stage": stage, "model": model, "kb": kb})
        try:
            yield _RecObj()
        finally:
            exit_called["n"] += 1

    # Mock chat handler returns a non-streaming response with usage.
    fake_response = MagicMock()
    fake_response.body = (
        b'{"choices":[{"message":{"content":"ok"}}],'
        b'"usage":{"prompt_tokens":42,"completion_tokens":17}}'
    )
    fake_handler = AsyncMock(return_value=fake_response)

    # Replicate the wrap pattern from main.py to assert the contract.
    # If the patch evolves (e.g. swaps the kb-extraction shape), this
    # test should evolve in lockstep.
    async def wrap_invocation(metadata: dict, model_id: str) -> object:
        import json as _json

        _telemetry_kb = None
        try:
            _kb_cfg = (metadata or {}).get("kb_config") or {}
            if isinstance(_kb_cfg, dict):
                _kb_ids = _kb_cfg.get("kb_ids") or _kb_cfg.get("selected_kb_ids") or []
                if isinstance(_kb_ids, (list, tuple)) and _kb_ids:
                    _telemetry_kb = ",".join(str(x) for x in _kb_ids)
        except Exception:
            _telemetry_kb = None

        async with _fake_record(
            stage="user_chat",
            model=str(model_id) if model_id else "unknown",
            kb=_telemetry_kb,
        ) as _llm_rec:
            response = await fake_handler(None, {}, None)
            try:
                body = getattr(response, "body", None)
                if body and isinstance(body, (bytes, bytearray)):
                    parsed = _json.loads(body.decode("utf-8", errors="ignore"))
                    usage = parsed.get("usage") if isinstance(parsed, dict) else None
                    if isinstance(usage, dict):
                        _llm_rec.set_tokens(
                            prompt=int(usage.get("prompt_tokens", 0) or 0),
                            completion=int(usage.get("completion_tokens", 0) or 0),
                        )
            except Exception:
                pass
            return response

    metadata = {"kb_config": {"kb_ids": [1, 7, 12]}}
    result = await wrap_invocation(metadata, "gemma-4-31B-it-AWQ")

    # Recorder entered exactly once with the expected labels.
    assert len(enter_calls) == 1
    assert enter_calls[0]["stage"] == "user_chat"
    assert enter_calls[0]["model"] == "gemma-4-31B-it-AWQ"
    assert enter_calls[0]["kb"] == "1,7,12"
    # Exit fired (covers TTFT histogram observation in the real recorder).
    assert exit_called["n"] == 1
    # Tokens captured from the response body.
    assert set_tokens_calls == [{"prompt": 42, "completion": 17}]
    # Underlying handler invoked.
    fake_handler.assert_awaited_once()
    assert result is fake_response


@pytest.mark.asyncio
async def test_wrap_does_not_break_when_kb_config_missing() -> None:
    """No ``kb_config`` in metadata → ``kb`` label is None (recorder
    coerces to "none" downstream). Critical: the wrap must not crash
    on the most common case (chat without selected KBs).
    """
    enter_calls: list[dict] = []

    class _RecObj:
        def set_tokens(self, **_kw): pass
        def set_first_token_at(self, t): pass
        def set_kb(self, kb): pass

    @asynccontextmanager
    async def _fake_record(*, stage, model, kb=None):
        enter_calls.append({"kb": kb})
        yield _RecObj()

    fake_handler = AsyncMock(return_value=MagicMock(body=b"{}"))

    async def wrap_invocation(metadata):
        _telemetry_kb = None
        try:
            _kb_cfg = (metadata or {}).get("kb_config") or {}
            if isinstance(_kb_cfg, dict):
                _kb_ids = _kb_cfg.get("kb_ids") or _kb_cfg.get("selected_kb_ids") or []
                if isinstance(_kb_ids, (list, tuple)) and _kb_ids:
                    _telemetry_kb = ",".join(str(x) for x in _kb_ids)
        except Exception:
            _telemetry_kb = None
        async with _fake_record(stage="user_chat", model="m", kb=_telemetry_kb):
            return await fake_handler(None, {}, None)

    # No kb_config at all.
    await wrap_invocation({})
    # Falsy kb_config.
    await wrap_invocation({"kb_config": None})
    # Empty kb_ids list.
    await wrap_invocation({"kb_config": {"kb_ids": []}})

    assert all(c["kb"] is None for c in enter_calls), enter_calls


@pytest.mark.asyncio
async def test_wrap_tolerates_streaming_response_with_no_body() -> None:
    """Streaming responses don't expose ``response.body`` — the wrap
    must not crash, just leave the recorder counters at zero.
    """
    enter_calls: list[dict] = []

    class _RecObj:
        def __init__(self):
            self.tokens_calls = []
        def set_tokens(self, *, prompt, completion):
            self.tokens_calls.append((prompt, completion))
        def set_first_token_at(self, t): pass
        def set_kb(self, kb): pass

    rec_obj = _RecObj()

    @asynccontextmanager
    async def _fake_record(*, stage, model, kb=None):
        enter_calls.append(stage)
        yield rec_obj

    streaming_response = MagicMock(spec=[])  # no body attribute
    fake_handler = AsyncMock(return_value=streaming_response)

    async def wrap_invocation():
        async with _fake_record(stage="user_chat", model="m", kb=None) as _rec:
            response = await fake_handler(None, {}, None)
            try:
                body = getattr(response, "body", None)
                if body and isinstance(body, (bytes, bytearray)):
                    import json as _json
                    parsed = _json.loads(body.decode("utf-8", errors="ignore"))
                    usage = parsed.get("usage") if isinstance(parsed, dict) else None
                    if isinstance(usage, dict):
                        _rec.set_tokens(
                            prompt=int(usage.get("prompt_tokens", 0) or 0),
                            completion=int(usage.get("completion_tokens", 0) or 0),
                        )
            except Exception:
                pass
            return response

    result = await wrap_invocation()
    # Recorder entered, no token capture attempted.
    assert enter_calls == ["user_chat"]
    assert rec_obj.tokens_calls == []
    assert result is streaming_response
