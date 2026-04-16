import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_pyproject_has_project_name():
    content = (ROOT / "pyproject.toml").read_text()
    assert 'name = "org-chat-assistant"' in content

def test_pytest_asyncio_installed():
    import pytest_asyncio  # noqa: F401

def test_sqlalchemy_async_installed():
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401

def test_testcontainers_installed():
    import testcontainers  # noqa: F401

def test_httpx_installed():
    import httpx  # noqa: F401

def test_argon2_installed():
    import argon2  # noqa: F401
