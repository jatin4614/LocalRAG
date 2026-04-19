"""Top-level conftest for pytest configuration.

Adds --integration flag that opts in to tests/integration/ (excluded by default
because those tests spin up docker compose, taking ~10 min and disrupting any
locally-running stack).
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Also run tests/integration/ (spins up docker compose — slow, heavy).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """When --integration is passed, extend testpaths to include tests/integration/.

    pyproject.toml pins testpaths=["tests/unit"] so naive `pytest` is unit-only.
    This hook broadens that when the opt-in flag is set, so `pytest --integration`
    (no path) picks up both suites. If the user already gave an explicit path,
    pytest uses that and ignores testpaths, so no change is needed for that case.
    """
    markexpr = config.getoption("-m", default="") or ""
    wants_integration = (
        config.getoption("--integration")
        or "integration" in markexpr
    )
    if not wants_integration:
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
    # Strategy: if user passed --integration OR explicitly asked for integration
    # paths / marker, let things through. Otherwise, skip anything under
    # tests/integration/ or marked @pytest.mark.integration.
    if config.getoption("--integration"):
        return
    # If the user directly pointed pytest at tests/integration/, honor it.
    # Heuristic: any arg in config.args contains "integration".
    args_mention_integration = any("integration" in str(a) for a in (config.args or []))
    if args_mention_integration:
        return
    # If the user used -m integration, collection already filtered by marker.
    markexpr = config.getoption("-m", default="") or ""
    if "integration" in markexpr:
        return
    # Otherwise, skip integration tests at collection time.
    skip_integration = pytest.mark.skip(reason="integration tests skipped (pass --integration to opt in)")
    for item in items:
        # Skip by path
        if "tests/integration" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip_integration)
        # Skip by marker
        elif any(m.name == "integration" for m in item.iter_markers()):
            item.add_marker(skip_integration)
