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
5 auth tests in tests/unit/ have been failing since before this branch (JWT/stub mode); they are out of scope for the RAG upgrade. Count them in your baseline.
