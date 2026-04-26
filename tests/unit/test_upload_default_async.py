import importlib
import os


def test_rag_sync_ingest_default_is_zero(monkeypatch):
    """Plan B Phase 6.2 — default flipped from 1 to 0 (async via Celery)."""
    monkeypatch.delenv("RAG_SYNC_INGEST", raising=False)
    from ext.routers import upload
    importlib.reload(upload)
    assert upload.RAG_SYNC_INGEST is False, (
        "Plan B Phase 6.2: RAG_SYNC_INGEST default must be 0 (async via Celery)"
    )


def test_rag_sync_ingest_can_still_be_enabled(monkeypatch):
    monkeypatch.setenv("RAG_SYNC_INGEST", "1")
    from ext.routers import upload
    importlib.reload(upload)
    assert upload.RAG_SYNC_INGEST is True


def test_compose_celery_worker_no_profile_gate():
    import yaml
    import pathlib
    compose_file = pathlib.Path(__file__).resolve().parents[2] / \
        "compose" / "docker-compose.yml"
    compose = yaml.safe_load(compose_file.read_text())
    assert "celery-worker" in compose["services"]
    profiles = compose["services"]["celery-worker"].get("profiles", [])
    assert profiles == [], (
        "celery-worker must be in the default compose profile (no profile gate)."
    )
