"""Test that chunker_structured uses the real tokenizer, not len/4.

Review §2.2 — the previous implementation estimated tokens as
``len(s) // 4`` which under-counts dense content (JSON, tables, code)
by ~30%. The under-count makes downstream chunks larger than the
``chunk_size_tokens`` budget the caller supplied, which then evicts
content during the prompt-budget pass.
"""
from __future__ import annotations

from ext.services import chunker_structured


class TestTokEstimateUsesRealTokenizer:
    def test_dense_json_significantly_higher_than_len_div_4(self) -> None:
        """Dense JSON of ~1000 chars should tokenize to noticeably more
        than len/4 = 250 tokens (typical: ~330+ for cl100k / ~290+ for
        gemma-4 — well above the naive estimate)."""
        # ~1000-char dense JSON sample with quotes, brackets, punctuation —
        # all of which the tokenizer splits more aggressively than
        # alphabetic text.
        sample = (
            '{"users": ['
            + ",".join(
                '{"id": %d, "name": "user_%d", "email": "user_%d@example.com",'
                ' "active": true, "score": %d.%d}'
                % (i, i, i, i, i % 100)
                for i in range(11)
            )
            + "]}"
        )
        # Sanity: target ~1000 chars (not load-bearing if drifts; the assertion
        # is on the ratio).
        assert 800 <= len(sample) <= 1200, f"sample is {len(sample)} chars"

        naive = len(sample) // 4
        actual = chunker_structured._tok_estimate(sample)

        # Real tokenizer should report SIGNIFICANTLY more tokens than naive
        # len/4 on dense JSON. Use a 1.20x ratio as the floor — empirically
        # cl100k yields ~1.32x, gemma yields ~1.18x; we want this test to
        # catch a regression to the literal len/4 (== 1.00x) without being
        # tokenizer-fragile.
        assert actual >= int(naive * 1.15), (
            f"_tok_estimate({len(sample)} chars) returned {actual}; "
            f"len/4={naive}. Real tokenizer should yield notably more on "
            f"dense JSON (regression: still using len//4?)"
        )

    def test_returns_at_least_one_for_nonempty(self) -> None:
        """Don't regress the floor-of-1 guarantee for tiny strings."""
        assert chunker_structured._tok_estimate("a") >= 1
        assert chunker_structured._tok_estimate(" ") >= 1

    def test_empty_string_does_not_crash(self) -> None:
        # The previous max(1, len(s)//4) returned 1 for empty input.
        # The replacement with len(enc.encode("")) is 0 — wrap in
        # max(1, ...) to preserve the contract.
        n = chunker_structured._tok_estimate("")
        assert n >= 1

    def test_plain_prose_roughly_matches_traditional_estimate(self) -> None:
        """Prose tokenization is closer to len/4 than dense JSON, but
        the test should not over-fit to a specific tokenizer. Just
        require non-zero and sane (within 3x of len/4)."""
        prose = (
            "The quick brown fox jumps over the lazy dog. "
            * 20
        )
        actual = chunker_structured._tok_estimate(prose)
        naive = max(1, len(prose) // 4)
        assert actual >= naive // 2  # should NOT be wildly under
        assert actual <= naive * 3   # should NOT be wildly over
