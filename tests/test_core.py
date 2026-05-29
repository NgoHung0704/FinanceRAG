"""Pure-Python tests for the FinanceRAG core (no ML stack required).

Run:  python -m pytest tests/test_core.py   (or: python tests/test_core.py)
These cover the logic that does NOT need torch/faiss/openai: chunking,
score fusion, evaluation metrics, prompt construction, and data parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financerag_app.chunking import chunk_corpus, detect_table_regions  # noqa: E402
from financerag_app.evaluate import evaluate, ndcg_at_k, recall_at_k  # noqa: E402
from financerag_app.generator import Passage, build_context, build_messages  # noqa: E402
from financerag_app.retriever import fuse_scores  # noqa: E402


def test_ndcg_perfect_and_zero():
    rel = {"a": 1, "c": 1}
    assert ndcg_at_k(["a", "c", "b"], rel, k=10) == 1.0  # both relevant on top
    assert ndcg_at_k(["x", "y", "z"], rel, k=10) == 0.0  # none retrieved
    assert ndcg_at_k([], rel, k=10) == 0.0


def test_ndcg_order_matters():
    rel = {"a": 1}
    top = ndcg_at_k(["a", "b", "c"], rel, k=10)
    low = ndcg_at_k(["b", "c", "a"], rel, k=10)
    assert top > low > 0.0


def test_recall_and_evaluate():
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"d": 1}}
    results = {"q1": ["a", "x", "y"], "q2": ["d"]}
    assert recall_at_k(results["q1"], qrels["q1"], k=10) == 0.5
    agg = evaluate(results, qrels, k=10)
    assert agg["num_queries"] == 2
    assert 0.0 < agg["NDCG@10"] <= 1.0


def test_fuse_scores_alpha_extremes():
    dense = {0: 1.0, 1: 0.0}
    sparse = {0: 0.0, 1: 1.0}
    # alpha=1 -> dense dominates (index 0 first)
    assert fuse_scores(dense, sparse, alpha=1.0)[0][0] == 0
    # alpha=0 -> sparse dominates (index 1 first)
    assert fuse_scores(dense, sparse, alpha=0.0)[0][0] == 1
    # balanced -> tie, both present
    assert len(fuse_scores(dense, sparse, alpha=0.5)) == 2


def test_fuse_scores_union_of_candidates():
    fused = fuse_scores({0: 1.0}, {5: 1.0}, alpha=0.5)
    idxs = {i for i, _ in fused}
    assert idxs == {0, 5}


def test_detect_tables():
    text = "Intro line\n| a | b | c |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\nOutro line"
    regions = detect_table_regions(text)
    assert regions, "should detect at least one table region"


def test_chunk_corpus_ids_and_mapping():
    corpus = [
        {"_id": "doc1", "title": "T", "text": "A" * 50},
        {"_id": "doc2", "title": "", "text": "word " * 400},
    ]
    chunks, mapping = chunk_corpus(corpus, method="recursive", size=200, overlap=20)
    assert chunks, "expected chunks"
    for c in chunks:
        assert c["_id"].startswith(c["original_id"])
        assert mapping[c["_id"]] == c["original_id"]
    # the long doc must produce more than one chunk
    assert sum(1 for c in chunks if c["original_id"] == "doc2") > 1


def test_chunk_none_method_keeps_whole_doc():
    corpus = [{"_id": "d", "title": "Apple", "text": "Revenue grew."}]
    chunks, _ = chunk_corpus(corpus, method="none", size=100, overlap=0)
    assert len(chunks) == 1
    assert "Apple" in chunks[0]["text"]


def test_build_context_numbers_passages():
    passages = [Passage("d1", "first", "T1"), Passage("d2", "second", "")]
    ctx = build_context(passages)
    assert "[1]" in ctx and "[2]" in ctx
    assert "d1" in ctx and "d2" in ctx


def test_build_messages_structure():
    msgs = build_messages("What is revenue?", [Passage("d1", "Revenue is 10.", "T")])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "What is revenue?" in msgs[1]["content"]
    assert "[1]" in msgs[1]["content"]


def test_lightrag_parse_doc_ids_order_dedup_limit():
    from financerag_app.lightrag_retriever import parse_doc_ids

    ctx = "...DOCID::doc9:: text ...DOCID::doc3:: more ...DOCID::doc9:: again ...DOCID::doc1::"
    assert parse_doc_ids(ctx, top_k=10) == ["doc9", "doc3", "doc1"]   # order + dedup
    assert parse_doc_ids(ctx, top_k=2) == ["doc9", "doc3"]            # top_k limit
    assert parse_doc_ids("", top_k=5) == []                          # empty safe


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
