import pytest

from ext.services.chunker_structured import chunk_structured


def test_fenced_code_block_emitted_as_single_chunk():
    text = """## Setup

```python
import asyncio
async def main():
    await asyncio.sleep(0)
```

That's the setup."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 1
    assert "import asyncio" in code_chunks[0]["text"]
    assert code_chunks[0].get("language") == "python"


def test_multiple_code_blocks_each_atomic():
    text = """First block.

```sh
echo hello
```

Middle text.

```python
print("hello")
```

End."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 2
    languages = sorted(c.get("language") for c in code_chunks)
    assert languages == ["python", "sh"]


def test_code_block_without_language_tagged_unknown():
    text = """```
just some text
```"""
    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 1
    assert code_chunks[0].get("language") in ("", None, "unknown", "text")


def test_inline_code_not_treated_as_code_chunk():
    text = """A paragraph with `inline_code` mixed in. Just prose."""
    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 0


def test_oversized_code_block_split_with_continuation_marker():
    huge = "x = 1\n" * 500
    text = f"""```python
{huge}```"""
    chunks = chunk_structured(text, chunk_size_tokens=200, overlap_tokens=20)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) > 1
    # Each chunk must announce continuation in payload
    for cc in code_chunks[1:]:
        assert cc.get("continuation") is True
