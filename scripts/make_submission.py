"""Generate the competition submission CSV (query_id, corpus_id).

Runs the full hybrid pipeline (retrieve -> rerank) over ALL queries of each
dataset and writes the top-10 documents per query in the competition format:

    query_id,corpus_id
    q00001,MSFT20230014
    ... (10 rows per query)

Usage:
    python scripts/make_submission.py                      # all datasets -> submission.csv
    python scripts/make_submission.py finder tatqa         # subset
    python scripts/make_submission.py -o my_submission.csv --top-k 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financerag_app import data as data_mod  # noqa: E402
from financerag_app.config import DATASETS, AppConfig  # noqa: E402
from financerag_app.data import dataset_available  # noqa: E402
from financerag_app.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the FinanceRAG submission CSV")
    ap.add_argument("datasets", nargs="*", help="datasets (default: all available)")
    ap.add_argument("-o", "--output", default="submission.csv", help="output CSV path")
    ap.add_argument("--top-k", type=int, default=10, help="docs per query (default 10)")
    args = ap.parse_args()

    cfg = AppConfig()
    cfg.use_generation = False                 # retrieval only — submission needs ranked docs
    pipe = RAGPipeline(cfg)
    targets = [d.lower() for d in args.datasets] or list(DATASETS)

    rows = [("query_id", "corpus_id")]
    total_q = 0
    for ds in targets:
        ok, msg = dataset_available(ds, cfg.data_dir)
        if not ok:
            print(f"  ⏭  {ds:14s} skipped ({msg})")
            continue
        queries = data_mod.load_queries(ds, cfg.data_dir)
        print(f"  ▶  {ds:14s} {len(queries)} queries ...", flush=True)
        results = pipe.batch_retrieve(ds, queries, top_k=args.top_k)   # {qid: [doc_id,...]}
        for qid, docs in results.items():
            for cid in docs[: args.top_k]:
                rows.append((qid, cid))
        total_q += len(results)

    out = Path(args.output)
    with open(out, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"\n✅ Wrote {out}  ({len(rows) - 1:,} rows, {total_q:,} queries)")
    # quick sanity
    per_q = {}
    for qid, _ in rows[1:]:
        per_q[qid] = per_q.get(qid, 0) + 1
    counts = sorted(set(per_q.values()))
    print(f"   docs/query seen: {counts}  (expected [{args.top_k}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
