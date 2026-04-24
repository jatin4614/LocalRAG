from pathlib import Path
from tests.eval.seed_test_kb import collect_corpus_docs


def test_collects_all_year_buckets():
    corpus_dir = Path(__file__).resolve().parents[1] / "eval" / "seed_corpus"
    docs = collect_corpus_docs(corpus_dir)
    years = {d["year_bucket"] for d in docs}
    assert years == {"2023", "2024", "2025", "2026"}


def test_deterministic_doc_ids():
    corpus_dir = Path(__file__).resolve().parents[1] / "eval" / "seed_corpus"
    docs1 = collect_corpus_docs(corpus_dir)
    docs2 = collect_corpus_docs(corpus_dir)
    ids1 = sorted(d["doc_id"] for d in docs1)
    ids2 = sorted(d["doc_id"] for d in docs2)
    assert ids1 == ids2, "doc_id must be deterministic across runs"
