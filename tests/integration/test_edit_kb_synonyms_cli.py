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


def test_add_merges_into_existing(env):
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


def test_add_returns_clear_error_on_missing_kb(env):
    """--add on a nonexistent KB exits 2 with an error message."""
    r = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "99999", "--add", '["x"]'],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "no KB" in r.stderr.lower() or "99999" in r.stderr


def test_load_rejects_non_string_inner_elements(env, tmp_path):
    """--load with bad shape exits 2 without writing."""
    bad = tmp_path / "bad.json"
    bad.write_text('[[1, 2], ["valid", "shape"]]')
    r = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "2", "--load", str(bad)],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "strings" in r.stderr.lower() or "string" in r.stderr.lower()


def test_set_returns_clear_error_on_missing_kb(env, tmp_path):
    """--load on a nonexistent KB exits 2 (was: silent success)."""
    good = tmp_path / "good.json"
    good.write_text('[]')
    r = subprocess.run(
        ["./.venv/bin/python", SCRIPT, "--kb", "99999", "--load", str(good)],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 2
    assert "no KB" in r.stderr.lower() or "99999" in r.stderr
