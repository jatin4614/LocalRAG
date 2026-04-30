"""Static analysis of ``ext/static/kb-admin.html``.

These tests pin the security and contract properties of the standalone
admin UI HTML. They run as ordinary unit tests (no browser, no network)
by reading the file as text and grepping with regex.

Coverage:

* C1.a — No raw template-literal interpolation of unescaped variables
  into ``innerHTML``. The guarded form is always
  ``innerHTML = ''`` (literal empty-string assignment used to wipe a
  node before re-rendering, allowed because the assigned value is a
  constant). Dynamic content goes through the safe-DOM helpers (``el``,
  ``textContent``, ``appendChild``).

* C1.b — The file upload allowlist includes ``.pptx`` (H11).

* C1.c — ``localStorage.token`` is not used. The previous direct
  ``localStorage.token`` slot read is replaced with a one-shot
  ``getItem('token')`` followed immediately by ``removeItem('token')``
  — the JWT lives in a module-scoped const for the rest of the page.

A future reviewer who reverts to ``el.innerHTML = `${untrusted}…``` or
brings back ``localStorage.token`` will be caught by these tests, not
by a Friday-evening prod XSS report.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

HTML_PATH = Path(__file__).resolve().parents[2] / "ext" / "static" / "kb-admin.html"


@pytest.fixture(scope="module")
def html_text() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


def test_html_file_exists_and_is_nonempty() -> None:
    assert HTML_PATH.exists(), f"missing {HTML_PATH}"
    assert HTML_PATH.stat().st_size > 0, "kb-admin.html unexpectedly empty"


def test_no_unsafe_innerhtml_template_interpolation(html_text: str) -> None:
    """C1.a — no ``innerHTML = `…${var}…``` patterns.

    Detect the high-risk shape: an ``innerHTML`` assignment whose RHS is
    a JavaScript template literal containing at least one ``${…}``
    placeholder. Any such placeholder is a likely XSS sink unless every
    interpolation is HTML-escaped, and we don't escape — so the test
    rejects all of them. (The safe-DOM helpers in the file replace this
    pattern entirely; if a future change brings it back we want a
    failing test, not a Friday-evening prod incident.)

    The ``innerHTML = ''`` literal-empty-string assignment used to wipe
    a container before re-rendering is not template-interpolation and
    is therefore allowed. Likewise, ``document.body.innerHTML = '';``
    in the logged-out splash path is a literal empty-string and OK.
    """
    # Pattern: `innerHTML` (any whitespace) `=` (any whitespace) backtick
    # followed by anything-up-to a `${…}` placeholder followed by anything
    # then a closing backtick. Multiline because real-world templates
    # span lines.
    pattern = re.compile(
        r"\.innerHTML\s*=\s*`[^`]*\$\{[^}]+\}[^`]*`",
        re.DOTALL,
    )
    matches = pattern.findall(html_text)
    assert not matches, (
        f"unsafe innerHTML template-literal interpolation found "
        f"({len(matches)} occurrence(s)):\n"
        + "\n---\n".join(m[:240] for m in matches)
    )


def test_pptx_in_upload_allowlist(html_text: str) -> None:
    """H11 — ``.pptx`` must appear in the file ``accept=`` allowlist.

    The exact list is chosen so backend extractor.py support (which
    handles pptx via python-pptx) is exposed at the UI layer. We accept
    either the canonical superset or any list that contains ``.pptx``.
    """
    assert ".pptx" in html_text, "kb-admin.html missing .pptx in allowlist"
    # Stronger: the input element's accept attribute mentions it.
    accept_attrs = re.findall(r'accept="([^"]*)"', html_text)
    assert accept_attrs, "no <input ... accept='...'> attribute found"
    assert any(".pptx" in a for a in accept_attrs), (
        f"none of the accept= attributes include .pptx: {accept_attrs}"
    )


def test_localstorage_token_not_directly_read(html_text: str) -> None:
    """C1.c — ``localStorage.token`` is never accessed via dot notation.

    The previous code did ``const T = localStorage.token`` (L138),
    which is XSS-readable on every page load. The replacement is a
    one-shot ``getItem('token')`` + ``removeItem('token')`` so the slot
    is empty for any later attacker JS that runs on the page.

    This test forbids the dot-form access entirely. Any access to the
    JWT MUST go through the one-shot init pattern; a future regression
    would be flagged as soon as the test runs.
    """
    # ``localStorage.token`` directly anywhere in the script.
    assert "localStorage.token" not in html_text, (
        "localStorage.token usage detected — JWT must be read via "
        "localStorage.getItem('token') exactly once in the init path "
        "and the slot scrubbed with removeItem('token') (see C1)."
    )


def test_jwt_init_pattern_present(html_text: str) -> None:
    """C1.c — the one-shot init pattern is present.

    ``getItem('token')`` and ``removeItem('token')`` both appear so the
    file genuinely follows the init-and-scrub contract. Either name is
    insufficient on its own (a file could call only ``getItem`` and
    leave the JWT in localStorage forever).
    """
    assert "getItem('token')" in html_text or 'getItem("token")' in html_text, (
        "expected one-shot localStorage.getItem('token') for JWT init"
    )
    assert "removeItem('token')" in html_text or 'removeItem("token")' in html_text, (
        "expected localStorage.removeItem('token') to scrub the slot"
    )
