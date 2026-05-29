"""Compare the hybrid retriever vs LightRAG on NDCG@10 (retrieve-only).

Both systems retrieve over the *same* corpus (sampled for large datasets so
LightRAG's LLM graph-build stays cheap), then we score their top-10 doc lists
against the qrels with the shared harness — apples-to-apples retrieval.

Needs: `pip install lightrag-hku` and OPENAI_API_KEY (LightRAG builds its graph
with an LLM at ingest time — that is the cost). The hybrid side uses the local
embedding + (optional) BM25; no rerank, to isolate the retrieval step.

Examples:
    python scripts/compare_lightrag.py                       # finqabench, financebench, multiheirtt(sample)
    python scripts/compare_lightrag.py finqabench
    python scripts/compare_lightrag.py multiheirtt --sample 300 --max-queries 40
    python scripts/compare_lightrag.py financebench --rebuild --mode mix
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financerag_app import data as data_mod  # noqa: E402
from financerag_app.chunking import chunk_corpus  # noqa: E402
from financerag_app.config import AppConfig, get_dataset_config  # noqa: E402
from financerag_app.evaluate import ndcg_at_k, recall_at_k  # noqa: E402
from financerag_app.retriever import HybridRetriever  # noqa: E402

DEFAULT_DATASETS = ["finqabench", "financebench", "multiheirtt"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sample_corpus(corpus, qrels, judged, max_size, seed=42):
    """Keep every relevant doc for the judged queries, fill the rest at random."""
    relevant = set()
    for q in judged:
        relevant.update(qrels.get(q["_id"], {}).keys())
    by_id = {d["_id"]: d for d in corpus}
    kept = [by_id[d] for d in relevant if d in by_id]
    rest = [d for d in corpus if d["_id"] not in relevant]
    random.Random(seed).shuffle(rest)
    kept.extend(rest[: max(0, max_size - len(kept))])
    return kept


def eval_system(retrieve_fn, judged, qrels):
    ndcgs, recalls = [], []
    for q in judged:
        ids = retrieve_fn(q["text"])
        rel = qrels[q["_id"]]
        ndcgs.append(ndcg_at_k(ids, rel, 10))
        recalls.append(recall_at_k(ids, rel, 10))
    return _mean(ndcgs), _mean(recalls)


def run_dataset(cfg, dataset, sample, max_queries, mode, rebuild):
    qrels = data_mod.load_qrels(dataset, cfg.data_dir)
    if not qrels:
        print(f"{dataset}: no qrels, skipped")
        return None
    judged = [q for q in data_mod.load_queries(dataset, cfg.data_dir) if q["_id"] in qrels]
    if max_queries:
        judged = judged[:max_queries]

    corpus = data_mod.load_corpus(dataset, cfg.data_dir)
    full_n = len(corpus)
    if full_n > sample:
        corpus = sample_corpus(corpus, qrels, judged, sample)
    print(f"\n=== {dataset} === corpus={full_n} -> using {len(corpus)} docs | {len(judged)} judged queries")

    dc = get_dataset_config(dataset)

    # --- Hybrid (dense + BM25), retrieve-only ---
    print("  building hybrid index...")
    chunks, _ = chunk_corpus(corpus, dc.chunk_method, dc.chunk_size, dc.chunk_overlap)
    hr = HybridRetriever(cfg).build(chunks)

    def hybrid_retrieve(text):
        hits = hr.retrieve(text, top_k_retrieval=max(dc.top_k_retrieval, 100),
                           alpha=dc.hybrid_alpha, top_k_docs=10)
        return [h.doc_id for h in hits]

    hyb_ndcg, hyb_recall = eval_system(hybrid_retrieve, judged, qrels)

    # --- LightRAG (graph), retrieve-only ---
    from financerag_app.lightrag_retriever import LightRAGRetriever  # lazy (optional dep)

    tag = f"{dataset}_s{len(corpus)}"
    lr = LightRAGRetriever(cfg, tag, mode=mode)
    if rebuild and lr.working_dir.exists():
        shutil.rmtree(lr.working_dir)
    if not LightRAGRetriever.is_built(cfg, tag):
        print("  building LightRAG graph (LLM ingest — this costs API calls)...")
        lr.build(corpus)
    else:
        print("  reusing cached LightRAG graph")
    lr_ndcg, lr_recall = eval_system(lambda t: lr.retrieve(t, top_k=10), judged, qrels)

    return {"dataset": dataset, "hyb_ndcg": hyb_ndcg, "hyb_recall": hyb_recall,
            "lr_ndcg": lr_ndcg, "lr_recall": lr_recall}


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid vs LightRAG NDCG@10")
    ap.add_argument("datasets", nargs="*", default=[], help="datasets (default: small set)")
    ap.add_argument("--sample", type=int, default=300, help="max docs for large corpora")
    ap.add_argument("--max-queries", type=int, default=0, help="cap judged queries (0 = all)")
    ap.add_argument("--mode", default="hybrid", help="LightRAG mode: naive|local|global|hybrid|mix")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the LightRAG graph from scratch")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.use_generation = False
    targets = [d.lower() for d in args.datasets] or DEFAULT_DATASETS

    rows = []
    for ds in targets:
        try:
            r = run_dataset(cfg, ds, args.sample, args.max_queries, args.mode, args.rebuild)
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {ds} failed: {exc}")
            r = None
        if r:
            rows.append(r)

    if rows:
        print("\n" + "=" * 68)
        print(f"{'dataset':<14}{'Hybrid NDCG':>13}{'LightRAG NDCG':>15}{'  Δ':>8}   winner")
        print("-" * 68)
        for r in rows:
            d = r["lr_ndcg"] - r["hyb_ndcg"]
            winner = "LightRAG" if d > 0.01 else ("hybrid" if d < -0.01 else "~tie")
            print(f"{r['dataset']:<14}{r['hyb_ndcg']:>13.3f}{r['lr_ndcg']:>15.3f}{d:>+8.3f}   {winner}")
        print(f"\n(mode={args.mode}; retrieve-only, no rerank; LightRAG uses OpenAI embeddings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
