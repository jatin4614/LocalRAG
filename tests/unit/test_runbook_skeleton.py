from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "runbook"


def test_runbook_index_exists():
    assert (RUNBOOK / "README.md").exists()


def test_flag_reference_has_every_rag_flag():
    content = (RUNBOOK / "flag-reference.md").read_text()
    required_flags = [
        "RAG_HYBRID", "RAG_RERANK", "RAG_MMR", "RAG_CONTEXT_EXPAND",
        "RAG_SPOTLIGHT", "RAG_SEMCACHE", "RAG_HYDE", "RAG_RAPTOR",
        "RAG_CONTEXTUALIZE_KBS", "RAG_INTENT_ROUTING", "RAG_DISABLE_REWRITE",
        "RAG_SYNC_INGEST", "RAG_BUDGET_TOKENIZER", "RAG_RBAC_CACHE_TTL_SECS",
        "RAG_CIRCUIT_BREAKER_ENABLED", "RAG_TENACITY_RETRY",
    ]
    for f in required_flags:
        assert f in content, f"flag-reference.md missing entry for {f}"


def test_troubleshooting_has_sections():
    content = (RUNBOOK / "troubleshooting.md").read_text()
    for section in ["retrieval is slow", "retrieval returns empty", "rbac denial"]:
        assert section.lower() in content.lower(), \
            f"troubleshooting.md missing section about {section}"
