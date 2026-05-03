"""Smoke test for scripts/edit_kb_synonyms.py — verifies the CLI surface
and that --add merges into the existing JSONB array."""
import json
import os
import subprocess
import pytest


SCRIPT = "scripts/edit_kb_synonyms.py"


@pytest.fixture
def env():
    e = os.environ.copy()
    if "DATABASE_URL" not in e:
        pwd = open("compose/.env").read()
        for line in pwd.splitlines():
            if line.startswith("POSTGRES_PASSWORD="):
                pw = line.split("=", 1)[1]
                e["DATABASE_URL"] = (
                    f"postgresql://orgchat:{pw}@localhost:5432/orgchat"
                )
                break
    return e


def test_list_returns_jsonb_array(env):
    r = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
        env=env, capture_output=True, text=True, check=True,
    )
    parsed = json.loads(r.stdout.strip())
    assert isinstance(parsed, list)


def test_add_merges_into_existing(env, tmp_path):
    # Save current state
    r0 = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
        env=env, capture_output=True, text=True, check=True,
    )
    original = json.loads(r0.stdout.strip())

    test_class = ["__test_marker_5pok__", "__test_marker_5POK__"]
    try:
        subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2",
             "--add", json.dumps(test_class)],
            env=env, capture_output=True, text=True, check=True,
        )
        r2 = subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2", "--list"],
            env=env, capture_output=True, text=True, check=True,
        )
        after = json.loads(r2.stdout.strip())
        assert test_class in after
        assert len(after) == len(original) + 1
    finally:
        # Cleanup — remove the test class
        subprocess.run(
            ["./.venv/bin/python", SCRIPT, "--kb", "2",
             "--remove", json.dumps(test_class)],
            env=env, capture_output=True, text=True,
        )
