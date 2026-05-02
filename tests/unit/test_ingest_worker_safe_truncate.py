"""Bug-fix campaign §1.7 — UTF-8-safe truncation in ingest worker.

The async ingest path used ``f"...{exc}"[:500]`` which splits multi-byte
codepoints when the underlying ``str.__getitem__`` slice happens to land
on a surrogate or extended grapheme. Worse, the upload (sync) path uses
``_safe_truncate`` which enforces a byte budget — the two paths produced
inconsistent error messages on identical errors.

The fix shares a single helper, ``_safe_truncate``, that:
1. Encodes to UTF-8.
2. Truncates by byte budget.
3. Decodes back with ``errors='ignore'`` so a mid-codepoint cut doesn't
   produce mojibake (the dangling partial bytes are silently dropped).
"""
from __future__ import annotations


def test_safe_truncate_short_string_unchanged():
    from ext.workers.ingest_worker import _safe_truncate
    assert _safe_truncate("hello", 500) == "hello"


def test_safe_truncate_ascii_truncated_to_byte_budget():
    from ext.workers.ingest_worker import _safe_truncate
    s = "x" * 1000
    out = _safe_truncate(s, 500)
    assert len(out.encode("utf-8")) == 500


def test_safe_truncate_multibyte_no_mojibake():
    """Hindi script: each Devanagari char is 3 bytes in UTF-8.

    A naive slice that lands mid-codepoint would produce a U+FFFD
    replacement char or a UnicodeDecodeError on serialization. The
    helper must produce a clean string (every byte forms a complete
    codepoint, every codepoint is renderable).
    """
    from ext.workers.ingest_worker import _safe_truncate
    # 200 Devanagari letters → 600 bytes → at byte budget 500 we cut
    # mid-codepoint. The helper must drop the partial codepoint
    # cleanly, not return mojibake or raise.
    s = "नमस्ते" * 50  # ~6 chars × 50 = 300 chars × ~3 bytes ~= ~900 bytes
    out = _safe_truncate(s, 500)
    encoded = out.encode("utf-8")
    assert len(encoded) <= 500
    # Round-tripping must succeed without errors (no half codepoints).
    out.encode("utf-8").decode("utf-8")
    # No replacement char (U+FFFD) should sneak in.
    assert "�" not in out


def test_safe_truncate_cjk_no_mojibake():
    """CJK is 3 bytes per char in UTF-8."""
    from ext.workers.ingest_worker import _safe_truncate
    s = "测试错误消息" * 100
    out = _safe_truncate(s, 200)
    assert len(out.encode("utf-8")) <= 200
    out.encode("utf-8").decode("utf-8")
    assert "�" not in out


def test_safe_truncate_emoji_4byte():
    """Emoji is 4 bytes in UTF-8."""
    from ext.workers.ingest_worker import _safe_truncate
    s = "🎉" * 200  # 800 bytes
    out = _safe_truncate(s, 250)
    assert len(out.encode("utf-8")) <= 250
    out.encode("utf-8").decode("utf-8")
    assert "�" not in out


def test_safe_truncate_default_byte_budget():
    """Calling with no max_len uses the module-default budget."""
    from ext.workers.ingest_worker import _safe_truncate
    # Default is 500 (matches the previous in-line ``[:500]`` behaviour).
    s = "x" * 1000
    out = _safe_truncate(s)
    assert len(out.encode("utf-8")) == 500
