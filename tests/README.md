# Running tests

## Unit tests (default)
Fast, no Docker. Runs in ~2-3 seconds.

    pytest                              # runs tests/unit/
    pytest tests/unit/test_chunker.py   # explicit file

## Integration tests (opt-in)
Spin up Postgres + Qdrant + Redis via docker compose. Runs in ~10 minutes.
Disrupts any locally-running `orgchat` stack.

    pytest --integration                # runs unit + integration
    pytest tests/integration/           # explicit path (bypasses default filter)
    pytest -m integration               # marker-based opt-in

## Known pre-existing failures
**Updated 2026-05-02 (bug-fix campaign Wave 2 §9.5).** The "5 auth tests"
claim is now stale — those tests are green. Currently quarantined as
`@pytest.mark.xfail(strict=False)`:

- `tests/unit/test_image_caption_extraction.py::test_image_caption_emitted_for_pdf_image` — fake-PDF + RAG_VISION_RASTER_MIN_BYTES=5000 default mismatch.
- `tests/unit/test_time_decay_intent_gating.py::TestApplyTimeDecayToHits::test_decays_hit_by_shard_key_age` — expected-value drift (asserts 0.25 ± 0.05; impl returns ~0.31).

Both are in the bug-fix campaign Wave 4 backlog. Removing the xfail
decorator after fixing them is the acceptance criterion.
