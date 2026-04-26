import pytest

from ext.services.chunker_structured import chunk_structured


def test_pipe_table_emitted_as_single_chunk():
    text = """## Q1 budget

Some prose before.

| Quarter | Budget | Actual |
|---------|--------|--------|
| Q1      | 100    | 95     |
| Q2      | 110    | 105    |
| Q3      | 105    | 110    |

Some prose after the table."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    # Table includes header + all 3 rows
    body = table_chunks[0]["text"]
    assert "Q1" in body and "Q2" in body and "Q3" in body
    assert "| Quarter" in body  # header preserved


def test_html_table_emitted_as_single_chunk():
    text = """<p>some prose</p>
<table>
  <thead><tr><th>Q</th><th>Budget</th></tr></thead>
  <tbody>
    <tr><td>Q1</td><td>100</td></tr>
    <tr><td>Q2</td><td>110</td></tr>
  </tbody>
</table>
<p>more prose</p>"""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    assert "<table>" in table_chunks[0]["text"] or \
        "Q1" in table_chunks[0]["text"]


def test_giant_table_split_with_repeated_header():
    rows = "\n".join(
        f"| Row{i} | {i * 100} | {i * 95} |"
        for i in range(500)  # ~3 KB+ table
    )
    text = f"""prose

| Header | A | B |
|--------|---|---|
{rows}

trailer prose"""

    chunks = chunk_structured(text, chunk_size_tokens=400, overlap_tokens=0)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) > 1, "table > limit must be split"
    # Each split must repeat the header for context
    for tc in table_chunks:
        assert "| Header" in tc["text"]


def test_prose_around_table_still_window_chunked():
    text = """prose paragraph one. """ + (" word" * 1000) + """

| Header | A |
|--------|---|
| 1      | 2 |

prose paragraph after. """ + (" word" * 1000)

    chunks = chunk_structured(text, chunk_size_tokens=400, overlap_tokens=50)
    prose_chunks = [c for c in chunks if c["chunk_type"] == "prose"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(prose_chunks) >= 2  # before + after
    assert len(table_chunks) == 1
