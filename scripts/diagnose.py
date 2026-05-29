"""Diagnose whether poor results come from RETRIEVAL or from RANKING.

For each dataset it reports, over the judged queries:

* Recall@100 (retriever only, no rerank) — the *ceiling*: is the relevant doc
  even in the candidate pool?
* Recall@10 (retriever only) — how good is the raw hybrid ranking?
* NDCG@10 / Recall@10 (after the cross-encoder reranker) — the final result.

Reading the verdict:
* High Recall@100 but low NDCG@10  -> RANKING problem. The doc is found but
  ordered poorly; fix the reranker / fusion alpha (cheap). A new retrieval
  paradigm like LightRAG will NOT help here.
* Low Recall@100                   -> RETRIEVAL problem. The doc is missed
  entirely; better embeddings / query handling / a different paradigm may help.

No LLM/API calls — only the embedding + reranker models. Run e.g.:
    python scripts/diagnose.py
    python scripts/diagnose.py multiheirtt tatqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financerag_app import data as data_mod  # noqa: E402
from financerag_app.config import DATASETS, AppConfig  # noqa: E402
from financerag_app.evaluate import ndcg_at_k, recall_at_k  # noqa: E402
from financerag_app.pipeline import RAGPipeline  # noqa: E402


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def diagnose_dataset(pipe: RAGPipeline, dataset: str, depth: int, max_queries: int):
    qrels = data_mod.load_qrels(dataset, pipe.cfg.data_dir)
    if not qrels:
        return None
    queries = [q for q in data_mod.load_queries(dataset, pipe.cfg.data_dir) if q["_id"] in qrels]
    if max_queries:
        queries = queries[:max_queries]

    rec100, rec10_raw, rec10_rr, ndcg10_rr = [], [], [], []
    for q in queries:
        rel = qrels[q["_id"]]
        cand = pipe.retrieve_only(dataset, q["text"], depth=depth)   # no rerank
        rec100.append(recall_at_k(cand, rel, depth))
        rec10_raw.append(recall_at_k(cand, rel, 10))

        reranked = pipe._retrieve_and_rerank(dataset, q["text"], 10, alpha=None)  # with rerank
        ranked_ids = [p.doc_id for p in reranked]
        rec10_rr.append(recall_at_k(ranked_ids, rel, 10))
        ndcg10_rr.append(ndcg_at_k(ranked_ids, rel, 10))

    return {
        "n": len(queries),
        f"recall@{depth}_raw": _mean(rec100),
        "recall@10_raw": _mean(rec10_raw),
        "recall@10_rerank": _mean(rec10_rr),
        "ndcg@10_rerank": _mean(ndcg10_rr),
    }


def verdict(m, depth):
    ceiling = m[f"recall@{depth}_raw"]
    final = m["ndcg@10_rerank"]
    if ceiling < 0.5:
        return "RETRIEVAL-bound (relevant docs missed) -> paradigm/embeddings"
    if ceiling - final > 0.25:
        return "RANKING-bound (found but mis-ordered) -> fix reranker/alpha"
    return "balanced"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose retrieval vs ranking")
    ap.add_argument("datasets", nargs="*", help="datasets (default: all with qrels)")
    ap.add_argument("--depth", type=int, default=100, help="retrieval ceiling depth")
    ap.add_argument("--max-queries", type=int, default=0, help="cap judged queries per dataset (0 = all)")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.use_generation = False
    pipe = RAGPipeline(cfg)
    targets = [d.lower() for d in args.datasets] or list(DATASETS)
    d = args.depth

    header = f"{'dataset':<14}{'n':>5}{f'  Rec@{d}':>10}{'  Rec@10':>9}{'  Rec@10rr':>11}{'  NDCG@10':>10}   verdict"
    print(header)
    print("-" * len(header))
    for ds in targets:
        m = diagnose_dataset(pipe, ds, d, args.max_queries)
        if m is None:
            print(f"{ds:<14}    -   (no qrels, skipped)")
            continue
        print(
            f"{ds:<14}{m['n']:>5}{m[f'recall@{d}_raw']:>10.3f}{m['recall@10_raw']:>9.3f}"
            f"{m['recall@10_rerank']:>11.3f}{m['ndcg@10_rerank']:>10.3f}   {verdict(m, d)}"
        )
    print("\nRec@%d = retrieval ceiling (no rerank) | rr = after reranker" % d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
