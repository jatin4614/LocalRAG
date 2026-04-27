"""B3 — verify the shadow-mode JSONL file handler is installed correctly.

Soak found that the ``orgchat.qu_shadow`` logger only emitted to stderr
/ docker logs. ``scripts/analyze_shadow_log.py`` is the operator-facing
analyzer and it needs a persistent JSONL file. ``install_shadow_log_file_handler``
attaches a ``RotatingFileHandler``; ``ext/app.py`` calls it from the
FastAPI lifespan when ``RAG_QU_SHADOW_MODE=1``.
"""
from __future__ import annotations

import json
import logging
import logging.handlers


def _shadow_logger() -> logging.Logger:
    return logging.getLogger("orgchat.qu_shadow")


def _strip_handlers(logger: logging.Logger) -> None:
    """Remove any handlers we previously installed so each test starts clean."""
    from ext.services.query_intent import _SHADOW_LOG_HANDLER_SENTINEL

    for h in list(logger.handlers):
        if getattr(h, _SHADOW_LOG_HANDLER_SENTINEL, False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


def test_install_attaches_rotating_file_handler(tmp_path, monkeypatch):
    """When called, the helper installs a RotatingFileHandler with the
    bare-message formatter and the configured path."""
    from ext.services.query_intent import install_shadow_log_file_handler

    log_path = tmp_path / "qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", str(log_path))

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        handler = install_shadow_log_file_handler()
        assert handler is not None
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        # Bare-message formatter (the JSON is already serialized by
        # _emit_shadow_log; we don't want logger metadata mangling it).
        assert handler.formatter._fmt == "%(message)s"
        # Handler is attached to the shadow logger
        assert handler in logger.handlers
        # 50 MB cap, 5 backups (the agreed rotation policy)
        assert handler.maxBytes == 50 * 1024 * 1024
        assert handler.backupCount == 5
    finally:
        _strip_handlers(logger)


def test_install_writes_formatted_jsonl(tmp_path, monkeypatch):
    """End-to-end: emit a shadow log line, verify it lands in the file
    as a bare JSON object (no logger prefix)."""
    from ext.services import query_intent as qi

    log_path = tmp_path / "qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", str(log_path))

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        handler = qi.install_shadow_log_file_handler()
        assert handler is not None

        # Mimic _emit_shadow_log directly — it just dumps a JSON string.
        payload = {
            "query": "what changed last quarter",
            "regex_label": "specific",
            "llm_label": "global",
            "agree": False,
        }
        logger.info(json.dumps(payload))
        handler.flush()

        contents = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(contents) == 1
        # Must be parseable JSON, no logger prefix bleeding in.
        parsed = json.loads(contents[0])
        assert parsed["regex_label"] == "specific"
        assert parsed["agree"] is False
    finally:
        _strip_handlers(logger)


def test_install_is_idempotent(tmp_path, monkeypatch):
    """Calling install twice attaches exactly one handler."""
    from ext.services.query_intent import (
        _SHADOW_LOG_HANDLER_SENTINEL,
        install_shadow_log_file_handler,
    )

    log_path = tmp_path / "qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", str(log_path))

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        first = install_shadow_log_file_handler()
        second = install_shadow_log_file_handler()
        assert first is second
        sentinel_handlers = [
            h for h in logger.handlers
            if getattr(h, _SHADOW_LOG_HANDLER_SENTINEL, False)
        ]
        assert len(sentinel_handlers) == 1
    finally:
        _strip_handlers(logger)


def test_install_skips_when_directory_unwritable(tmp_path, monkeypatch, caplog):
    """If the target directory can't be created (eg /proc, read-only fs),
    the helper logs a warning and returns None instead of crashing."""
    from ext.services.query_intent import install_shadow_log_file_handler

    # /proc is a virtual fs — mkdir will fail.
    bad_path = "/proc/forbidden_dir/qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", bad_path)

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        with caplog.at_level("WARNING", logger="ext.services.query_intent"):
            handler = install_shadow_log_file_handler()
        assert handler is None
    finally:
        _strip_handlers(logger)


def test_handler_attached_when_shadow_mode_on(tmp_path, monkeypatch):
    """The startup-gated helper attaches the handler when RAG_QU_SHADOW_MODE=1.

    ``maybe_install_shadow_log_file_handler`` is the env-gated entry
    point ``ext/app.py``'s ``build_app`` / ``build_ext_routers`` call
    from startup. We test it directly so we don't have to spin up
    Postgres / Qdrant / TEI just to validate the gate.
    """
    from ext.services.query_intent import (
        _SHADOW_LOG_HANDLER_SENTINEL,
        maybe_install_shadow_log_file_handler,
    )

    log_path = tmp_path / "qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", str(log_path))

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        result = maybe_install_shadow_log_file_handler()
        assert result is not None, (
            "shadow handler must be installed when RAG_QU_SHADOW_MODE=1"
        )
        sentinel_handlers = [
            h for h in logger.handlers
            if getattr(h, _SHADOW_LOG_HANDLER_SENTINEL, False)
        ]
        assert len(sentinel_handlers) == 1
    finally:
        _strip_handlers(logger)


def test_handler_not_attached_when_shadow_mode_off(tmp_path, monkeypatch):
    """The startup-gated helper must NOT attach a handler when shadow is off
    — no point burning disk if the JSONL won't be analyzed."""
    from ext.services.query_intent import (
        _SHADOW_LOG_HANDLER_SENTINEL,
        maybe_install_shadow_log_file_handler,
    )

    log_path = tmp_path / "qu_shadow.jsonl"
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "0")
    monkeypatch.setenv("RAG_QU_SHADOW_LOG_PATH", str(log_path))

    logger = _shadow_logger()
    _strip_handlers(logger)
    try:
        result = maybe_install_shadow_log_file_handler()
        assert result is None, (
            "shadow handler must NOT be installed when RAG_QU_SHADOW_MODE=0"
        )
        sentinel_handlers = [
            h for h in logger.handlers
            if getattr(h, _SHADOW_LOG_HANDLER_SENTINEL, False)
        ]
        assert len(sentinel_handlers) == 0
    finally:
        _strip_handlers(logger)


def test_app_wires_maybe_install_in_startup_path():
    """``build_app`` and ``build_ext_routers`` must wire the gated
    installer into their startup sequence.

    We can't call ``build_app`` directly here — it requires real
    DATABASE_URL / Qdrant / TEI — so we assert the wiring contract by
    inspecting the source: every path that boots the app must reference
    ``maybe_install_shadow_log_file_handler``.
    """
    import inspect

    from ext import app as app_mod

    source = inspect.getsource(app_mod)
    assert source.count("maybe_install_shadow_log_file_handler") >= 2, (
        "both build_app and build_ext_routers must wire the gated "
        "shadow-log installer at startup; current count: "
        f"{source.count('maybe_install_shadow_log_file_handler')}"
    )
