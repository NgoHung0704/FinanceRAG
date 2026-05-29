"""Evaluate the retriever with NDCG@10 against the qrels (competition parity).

This reuses the exact retrieve+rerank path the app uses, so the interactive
system and the reported metric never drift apart.

Usage:
    python scripts/run_eval.py                 # all datasets with qrels
    python scripts/run_eval.py finder tatqa    # specific datasets
    python scripts/run_eval.py --no-rerank     # ablation: retrieval only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financerag_app import data as data_mod  # noqa: E402
from financerag_app.config import DATASETS, AppConfig  # noqa: E402
from financerag_app.evaluate import evaluate  # noqa: E402
from financerag_app.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FinanceRAG retrieval (NDCG@10)")
    parser.add_argument("datasets", nargs="*", help="datasets to evaluate (default: all)")
    parser.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder reranker")
    parser.add_argument("-k", type=int, default=10, help="cutoff k (default 10)")
    args = parser.parse_args()

    cfg = AppConfig()
    cfg.use_generation = False
    cfg.use_reranker = not args.no_rerank
    pipe = RAGPipeline(cfg)

    targets = [d.lower() for d in args.datasets] or list(DATASETS)
    rows = []
    weighted_sum = 0.0
    weighted_n = 0

    for ds in targets:
        qrels = data_mod.load_qrels(ds, cfg.data_dir)
        if not qrels:
            print(f"  ⏭  {ds:14s} no qrels — skipped")
            continue
        queries = data_mod.load_queries(ds, cfg.data_dir)
        # Only evaluate queries that have judgements.
        judged = [q for q in queries if q["_id"] in qrels]
        results = pipe.batch_retrieve(ds, judged, top_k=args.k)
        metrics = evaluate(results, qrels, k=args.k)
        rows.append((ds, metrics))
        weighted_sum += metrics[f"NDCG@{args.k}"] * metrics["num_queries"]
        weighted_n += metrics["num_queries"]
        print(
            f"  {ds:14s} NDCG@{args.k}={metrics[f'NDCG@{args.k}']:.4f}  "
            f"Recall@{args.k}={metrics[f'Recall@{args.k}']:.4f}  "
            f"MRR={metrics['MRR']:.4f}  (n={metrics['num_queries']})"
        )

    if weighted_n:
        print("-" * 64)
        print(f"  WEIGHTED AVERAGE NDCG@{args.k}: {weighted_sum / weighted_n:.4f}  "
              f"(rerank={'on' if cfg.use_reranker else 'off'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
