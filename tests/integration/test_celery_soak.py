"""Celery soak test integration wrapper.

Skipped by default. Operator runs explicitly:
  pytest -m integration tests/integration/test_celery_soak.py -v

Requires:
  - Celery worker running (compose service celery-worker)
  - Postgres + Redis healthy
  - RAG_SYNC_INGEST=0 in the open-webui environment
"""
from __future__ import annotations

import os
import subprocess
import time
import pytest

pytestmark = pytest.mark.integration


def test_celery_soak_1000_docs():
    token = os.environ.get("RAG_ADMIN_TOKEN")
    if not token:
        pytest.skip("RAG_ADMIN_TOKEN not set")

    # 1. Drive 1000 uploads
    upload = subprocess.run(
        [
            "python", "scripts/celery_soak_test.py",
            "--target-kb", "1", "--target-subtag", "1",
            "--doc-count", "1000", "--concurrency", "8",
        ],
        capture_output=True, text=True, timeout=1800,  # 30min cap
    )
    assert upload.returncode == 0, f"upload phase failed: {upload.stderr}"

    # 2. Wait for celery to drain
    print("Sleeping 5min for celery to process the queue...")
    time.sleep(300)

    # 3. Verify
    verify = subprocess.run(
        [
            "python", "scripts/celery_soak_test.py",
            "--verify", "--target-kb", "1",
            "--expected-doc-count", "1000",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert verify.returncode == 0, f"verification failed: {verify.stderr}"
