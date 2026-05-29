"""Data loading for FinanceRAG.

Uses only the standard library (``json``/``csv``) so it stays import-light and
unit-testable without pandas. The on-disk layout is the competition's:

    data/<name>_corpus.jsonl/corpus.jsonl     # {"_id","title","text"}
    data/<name>_queries.jsonl/queries.jsonl   # {"_id","title","text"}
    data/<Name>_qrels.tsv                      # query_id  corpus_id  score
    data/chunked_corpus/<name>_corpus_chunked_optimal.jsonl   # {"_id","original_id","text"}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .config import QRELS_FILENAMES

Document = Dict[str, str]
PathLike = Union[str, Path]


def load_jsonl(path: PathLike) -> List[dict]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def corpus_path(dataset: str, data_dir: PathLike) -> Path:
    return Path(data_dir) / f"{dataset}_corpus.jsonl" / "corpus.jsonl"


def queries_path(dataset: str, data_dir: PathLike) -> Path:
    return Path(data_dir) / f"{dataset}_queries.jsonl" / "queries.jsonl"


def qrels_path(dataset: str, data_dir: PathLike) -> Path:
    fname = QRELS_FILENAMES.get(dataset, f"{dataset}_qrels.tsv")
    return Path(data_dir) / fname


def load_corpus(dataset: str, data_dir: PathLike) -> List[Document]:
    """Load the raw corpus. Each doc keeps ``_id``, ``title``, ``text``."""
    docs = load_jsonl(corpus_path(dataset, data_dir))
    for d in docs:
        d["_id"] = str(d.get("_id", d.get("id", "")))
        d.setdefault("title", "")
        d.setdefault("text", "")
    return docs


def load_queries(dataset: str, data_dir: PathLike) -> List[Document]:
    """Load queries. Each query keeps ``_id`` and ``text``."""
    qs = load_jsonl(queries_path(dataset, data_dir))
    for q in qs:
        q["_id"] = str(q.get("_id", q.get("id", "")))
        q.setdefault("text", "")
    return qs


def load_qrels(dataset: str, data_dir: PathLike) -> Dict[str, Dict[str, int]]:
    """Load relevance judgements as ``{query_id: {corpus_id: score}}``.

    Returns an empty dict if the qrels file is missing. Handles both
    ``query_id``/``query-id`` column spellings.
    """
    path = qrels_path(dataset, data_dir)
    qrels: Dict[str, Dict[str, int]] = {}
    if not path.exists():
        return qrels

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        qcol = "query_id" if "query_id" in (reader.fieldnames or []) else "query-id"
        ccol = "corpus_id" if "corpus_id" in (reader.fieldnames or []) else "corpus-id"
        for row in reader:
            qid = str(row[qcol])
            cid = str(row[ccol])
            try:
                score = int(row.get("score", 1))
            except (TypeError, ValueError):
                score = 1
            qrels.setdefault(qid, {})[cid] = score
    return qrels


def load_prechunked(dataset: str, chunked_dir: PathLike) -> Optional[List[Document]]:
    """Load the optimal pre-chunked corpus, if present.

    Chunks carry ``_id`` (chunk id) and ``original_id`` (parent document id).
    Returns ``None`` when no pre-chunked file exists.
    """
    path = Path(chunked_dir) / f"{dataset}_corpus_chunked_optimal.jsonl"
    if not path.exists():
        return None
    chunks = load_jsonl(path)
    for c in chunks:
        c["_id"] = str(c.get("_id", c.get("chunk_id", "")))
        c["original_id"] = str(c.get("original_id", c.get("doc_id", c["_id"])))
        c.setdefault("text", "")
    return chunks


def doc_text(doc: Document, combine_title: bool = True) -> str:
    """Render a document/chunk to a single text string for embedding/BM25."""
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if combine_title and title and title not in text:
        return f"{title}. {text}" if text else title
    return text


def build_corpus_lookup(corpus: List[Document]) -> Dict[str, Document]:
    """Map document id -> document for O(1) text lookup at rerank/generation time."""
    return {d["_id"]: d for d in corpus}


def dataset_available(dataset: str, data_dir: PathLike) -> Tuple[bool, str]:
    """Quick check that a dataset's files exist; returns (ok, message)."""
    cp = corpus_path(dataset, data_dir)
    qp = queries_path(dataset, data_dir)
    if not cp.exists():
        return False, f"corpus missing: {cp}"
    if not qp.exists():
        return False, f"queries missing: {qp}"
    return True, "ok"
