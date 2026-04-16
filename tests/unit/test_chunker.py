from ext.services.chunker import chunk_text


def test_short_text_returns_one_chunk():
    chunks = chunk_text("hello world", chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].index == 0


def test_long_text_splits_with_overlap():
    para = ("word " * 2500).strip()
    chunks = chunk_text(para, chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) >= 3
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_text_returns_empty():
    assert chunk_text("", chunk_tokens=800, overlap_tokens=100) == []


def test_single_very_long_word_chunked():
    one_word = "a" * 8000
    chunks = chunk_text(one_word, chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) >= 1
