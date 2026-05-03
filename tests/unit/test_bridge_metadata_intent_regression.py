"""Regression for 2026-05-03 _do_decompose UnboundLocalError on metadata intent.

The per-entity rerank quota patch (commit f416dbe) referenced _do_decompose
on every code path, but _do_decompose was only assigned in the non-metadata
branch. Result: every metadata-intent query crashed with UnboundLocalError
and returned 0 KB sources.
"""
import os
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ext.services import chat_rag_bridge as br


@dataclass
class _FakeHybrid:
    """Minimal HybridClassification stub for the metadata-intent path."""
    intent: str = "metadata"
    resolved_query: str = "how many documents are in the KB"
    temporal_constraint: object = None
    entities: list = field(default_factory=list)
    confidence: float = 1.0
    source: str = "regex"
    escalation_reason: object = None
    regex_reason: str = "metadata:how_many_docs"
    cached: bool = False


@pytest.mark.asyncio
async def test_metadata_intent_does_not_crash_on_do_decompose():
    """_run_pipeline must not raise UnboundLocalError for _do_decompose.

    The metadata-intent path short-circuits retrieval (raw_hits = []) before
    the multi-entity decompose block that assigns _do_decompose.  The
    post-rerank quota check at line ~1964 references _do_decompose regardless
    of intent, so it must be initialized upfront.

    The silent-failure handler at line ~2295 swallows the UnboundLocalError
    and returns [], so this test asserts that _record_silent_failure was NOT
    called with stage "rag_pipeline" (which is the indicator that the crash
    happened and was hidden).
    """
    fake_vs = MagicMock()
    fake_emb = MagicMock()
    fake_session_factory = MagicMock()
    br.configure(
        vector_store=fake_vs, embedder=fake_emb, sessionmaker=fake_session_factory
    )

    # Provide a no-op async session context so catalog SQL doesn't fail hard.
    fake_session = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)
    fake_session.execute = AsyncMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    fake_session_factory.return_value = fake_session

    # Force RAG_INTENT_ROUTING=1 so the intent branch is active and the
    # metadata short-circuit at line ~1549 fires.
    env_overrides = {
        "RAG_INTENT_ROUTING": "1",
        "RAG_INJECT_DATETIME": "0",  # suppress datetime preamble SQL
    }

    failures_recorded: list[tuple[str, Exception]] = []

    original_rsf = br._record_silent_failure

    def _spy_rsf(stage: str, err: Exception) -> None:
        failures_recorded.append((stage, err))
        original_rsf(stage, err)

    with patch.dict(os.environ, env_overrides):
        with patch.object(
            br, "_classify_with_qu", AsyncMock(return_value=_FakeHybrid())
        ):
            with patch.object(br, "_record_silent_failure", side_effect=_spy_rsf):
                result = await br._run_pipeline(
                    query="how many documents are in the KB",
                    selected_kbs=[{"kb_id": 2, "subtag_ids": []}],
                    user_id="user-1",
                    chat_id=None,
                    progress_cb=None,
                )

    # The call must return a list (not raise).
    assert isinstance(result, list), f"expected list, got {type(result)}"

    # Critical: the 'rag_pipeline' stage must NOT appear in silent failures.
    # If it does, it means the UnboundLocalError (or another crash in the hot
    # path) was swallowed — the bug is present.
    pipeline_failures = [
        (s, e) for s, e in failures_recorded if s == "rag_pipeline"
    ]
    assert pipeline_failures == [], (
        "rag_pipeline silent failure recorded — the _do_decompose "
        f"UnboundLocalError is still present: {pipeline_failures}"
    )
