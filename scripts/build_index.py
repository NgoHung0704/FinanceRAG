"""Pre-build and cache retrieval indexes so the app starts fast.

Usage:
    python scripts/build_index.py                # all available datasets
    python scripts/build_index.py finder tatqa   # specific datasets
    python scripts/build_index.py --force        # rebuild even if cached
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from financerag_app.config import DATASETS, AppConfig  # noqa: E402
from financerag_app.data import dataset_available  # noqa: E402
from financerag_app.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FinanceRAG retrieval indexes")
    parser.add_argument("datasets", nargs="*", help="datasets to build (default: all available)")
    parser.add_argument("--force", action="store_true", help="rebuild even if a cache exists")
    args = parser.parse_args()

    cfg = AppConfig()
    pipe = RAGPipeline(cfg)
    targets = [d.lower() for d in args.datasets] or list(DATASETS)

    failures = []
    for ds in targets:
        ok, msg = dataset_available(ds, cfg.data_dir)
        if not ok:
            print(f"  ⏭  {ds:14s} skipped ({msg})")
            continue
        try:
            t0 = time.time()
            pipe.build_index(ds, force=args.force)
            print(f"  ✅ {ds:14s} built in {time.time() - t0:6.1f}s  -> {pipe._cache_dir(ds)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {ds:14s} failed: {exc}")
            failures.append(ds)

    if failures:
        print(f"\nDone with {len(failures)} failure(s): {failures}")
        return 1
    print("\nAll requested indexes are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
