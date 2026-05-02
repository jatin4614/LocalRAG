"""Top-level conftest for pytest configuration.

Integration tests run **by default** because they cover the security-boundary
suite (RBAC isolation, KB cross-user containment). Opt out for local-only
unit work with ``pytest --no-integration``.

History: Until 2026-05-02 (review §9.3) the polarity was inverted —
``--integration`` was an opt-in flag, which meant the isolation suite never
ran in default ``pytest`` invocations. The flip was a deliberate choice;
running these tests is non-negotiable per CLAUDE.md §2 invariant 1.

The legacy ``--integration`` flag is kept as an accepted no-op so existing
documentation, CI scripts, and Makefile targets continue to work.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def fake_redis():
    """In-memory async Redis double for cache unit tests.

    Implements just the subset of ``redis.asyncio.Redis`` that production
    code touches: ``get``, ``set`` (with ``ex`` TTL), ``ttl``, ``delete``.
    Returning a fresh instance per test gives each test isolation without
    monkey-patching the real client.
    """

    class _FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}
            self._ttls: dict[str, int] = {}

        async def get(self, key):
            return self._store.get(key)

        async def set(self, key, value, ex=None):
            self._store[key] = value
            if ex is not None:
                self._ttls[key] = int(ex)
            return True

        async def ttl(self, key):
            return self._ttls.get(key, -1)

        async def delete(self, key):
            self._store.pop(key, None)
            self._ttls.pop(key, None)
            return 1

    return _FakeRedis()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help=(
            "Skip tests/integration/ for fast local iteration. "
            "Default behaviour runs both unit and integration suites."
        ),
    )
    # Legacy: kept as a no-op so existing scripts that pass --integration
    # don't break. See module docstring.
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="(deprecated; integration runs by default — kept for backward compat)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Extend testpaths to include tests/integration/ unless --no-integration.

    pyproject.toml pins ``testpaths=["tests/unit"]`` so a naive ``pytest`` would
    only walk unit. This hook broadens that to include integration so the
    security-boundary suite runs by default. If the user gave an explicit path,
    pytest uses that and ignores testpaths, so no change is needed for that case.
    """
    if config.getoption("--no-integration"):
        return
    # Only extend testpaths if no explicit paths were passed on the CLI.
    # config.invocation_params.args holds the original CLI argv (positional + flags).
    cli_args = tuple(config.invocation_params.args)
    # Strip flag tokens; flags like -m integration or -k foo have a value token
    # that should NOT be treated as a positional path.
    positional: list[str] = []
    skip_next = False
    _VALUE_FLAGS = ("-m", "-k", "-p", "-W", "-o", "-c", "-r", "-n")
    for a in cli_args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            if a in _VALUE_FLAGS:
                skip_next = True
            continue
        positional.append(a)
    if positional:
        return
    existing = list(config.inipath and config.getini("testpaths") or [])
    if "tests/integration" not in existing:
        existing.append("tests/integration")
    # Re-assign config.args so pytest picks up the extended set.
    config.args = existing


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests **only** when --no-integration was passed.

    Default polarity: integration tests run. The security-boundary suite is
    mandatory CI per CLAUDE.md §2 invariant 1; we cannot make it opt-in.
    """
    if not config.getoption("--no-integration"):
        return
    # Honor explicit path requests even with --no-integration (e.g. someone
    # ran `pytest --no-integration tests/integration/test_kb_isolation.py`
    # — let it through, because they explicitly asked for integration).
    args_mention_integration = any("integration" in str(a) for a in (config.args or []))
    if args_mention_integration:
        return
    # Honor `-m integration` for the same reason.
    markexpr = config.getoption("-m", default="") or ""
    if "integration" in markexpr:
        return
    # Skip integration tests at collection time.
    skip_integration = pytest.mark.skip(
        reason="--no-integration: skipping integration suite (default is to run it)"
    )
    for item in items:
        # Skip by path
        if "tests/integration" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_integration)
        # Skip by marker
        elif any(m.name == "integration" for m in item.iter_markers()):
            item.add_marker(skip_integration)
