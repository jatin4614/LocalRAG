"""Per-KB contextualize opt-in (Phase 3.3).

The global ``RAG_CONTEXTUALIZE_KBS`` flag is too blunt for KB-mixed
deployments: some KBs (meeting notes, roadmaps, ADRs) benefit from per-
chunk LLM context augmentation while others (API docs, code) gain
nothing — and pay the chat-call cost regardless. ``should_contextualize``
lets a per-KB ``rag_config`` value explicitly override the env flag in
either direction:

* per-KB ``contextualize: false``  →  skip even when global is ``"1"``
* per-KB ``contextualize: true``   →  run even when global is ``"0"``
* per-KB key absent                 →  inherit the env flag

These four tests pin the precedence rules.
"""
from ext.services.ingest import should_contextualize


def test_global_flag_on_kb_override_off():
    """Global RAG_CONTEXTUALIZE_KBS=1 but per-KB rag_config says no → skip."""
    assert should_contextualize(
        env_flag="1",
        kb_rag_config={"contextualize": False},
    ) is False


def test_global_flag_off_kb_override_on():
    assert should_contextualize(
        env_flag="0",
        kb_rag_config={"contextualize": True},
    ) is True


def test_default_off_when_neither_set():
    assert should_contextualize(env_flag="0", kb_rag_config={}) is False


def test_default_on_when_global_set():
    assert should_contextualize(env_flag="1", kb_rag_config={}) is True
