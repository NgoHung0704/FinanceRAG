"""Corpus chunking for FinanceRAG.

The pipeline prefers the pre-chunked corpora under
``data/chunked_corpus/*_optimal.jsonl``. This module provides the on-the-fly
fallback when those files are absent, consolidating the table-aware logic from
the experiments.

Methods ``recursive``/``fixed``/``preserve_tables``/``none`` are pure-Python.
``semantic`` requires sentence-transformers; if unavailable it transparently
falls back to ``recursive`` so the rest of the stack keeps working.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .data import Document, doc_text

Chunk = Dict[str, str]

_MIN_CHUNK_CHARS = 80


# --------------------------------------------------------------------------- #
# Table detection (heuristic)
# --------------------------------------------------------------------------- #
def detect_table_regions(text: str) -> List[Tuple[int, int]]:
    """Return [start_line, end_line) ranges that look like tables."""
    lines = text.split("\n")
    regions: List[Tuple[int, int]] = []
    in_table = False
    start = 0
    for i, line in enumerate(lines):
        is_table = (
            line.count("|") >= 2
            or line.count("\t") >= 2
            or len(re.findall(r"\s{3,}", line)) >= 2
        )
        if is_table and not in_table:
            in_table, start = True, max(0, i - 1)
        elif not is_table and in_table:
            in_table = False
            end = min(len(lines), i + 1)
            if end - start >= 3:
                regions.append((start, end))
    if in_table:
        regions.append((start, len(lines)))
    return regions


# --------------------------------------------------------------------------- #
# Character chunking
# --------------------------------------------------------------------------- #
def _fixed_chunks(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    out: List[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        piece = text[i : i + size].strip()
        if len(piece) >= _MIN_CHUNK_CHARS:
            out.append(piece)
        if i + size >= len(text):
            break
    return out


def _recursive_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Split on progressively finer separators, then pack to ~size with overlap."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    units = [text]
    for sep in separators:
        if all(len(u) <= size for u in units):
            break
        nxt: List[str] = []
        for u in units:
            nxt.extend(u.split(sep) if len(u) > size else [u])
        units = nxt

    chunks: List[str] = []
    cur = ""
    for u in units:
        if not u:
            continue
        if len(cur) + len(u) + 1 <= size:
            cur = f"{cur} {u}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + " " + u).strip() if overlap and cur else u
            # if a single unit exceeds size, hard-split it
            while len(cur) > size:
                chunks.append(cur[:size].strip())
                cur = cur[size - overlap :] if overlap else cur[size:]
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if len(c) >= _MIN_CHUNK_CHARS] or ([text] if text.strip() else [])


def _preserve_table_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Keep detected tables intact; recursively chunk the surrounding prose."""
    if len(text) < size * 1.5:
        return [text] if text.strip() else []
    lines = text.split("\n")
    regions = detect_table_regions(text)
    if not regions:
        return _recursive_chunks(text, size, overlap)

    chunks: List[str] = []
    prev = 0
    for start, end in regions:
        if start > prev:
            before = "\n".join(lines[prev:start]).strip()
            if before:
                chunks.extend(_recursive_chunks(before, size, overlap))
        table = "\n".join(lines[start:end]).strip()
        if table:
            chunks.append(f"[TABLE]\n{table}")
        prev = end
    if prev < len(lines):
        after = "\n".join(lines[prev:]).strip()
        if after:
            chunks.extend(_recursive_chunks(after, size, overlap))
    return chunks


def _semantic_chunks(text: str, size: int, threshold: float) -> List[str]:
    """Group adjacent sentences by embedding similarity (lazy import).

    Falls back to recursive chunking if sentence-transformers is unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        import numpy as np  # noqa: WPS433
    except Exception:  # pragma: no cover - depends on optional heavy deps
        return _recursive_chunks(text, size, int(size * 0.1))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    model = _semantic_model()
    embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    chunks: List[str] = []
    cur = [sentences[0]]
    cur_len = len(sentences[0])
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i], embs[i - 1]))
        if sim >= threshold and cur_len + len(sentences[i]) <= size:
            cur.append(sentences[i])
            cur_len += len(sentences[i])
        else:
            chunks.append(" ".join(cur))
            cur, cur_len = [sentences[i]], len(sentences[i])
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


_SEMANTIC_MODEL = None


def _semantic_model():  # pragma: no cover - heavy
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def chunk_document(doc: Document, method: str, size: int, overlap) -> List[str]:
    """Return the list of chunk texts for a single document."""
    text = doc_text(doc, combine_title=True)
    if not text.strip():
        return []
    if method == "none":
        return [text]
    if method == "fixed":
        return _fixed_chunks(text, size, int(overlap))
    if method == "preserve_tables":
        return _preserve_table_chunks(text, size, int(overlap))
    if method == "semantic":
        return _semantic_chunks(text, size, float(overlap))
    # default
    return _recursive_chunks(text, size, int(overlap))


def chunk_corpus(corpus: List[Document], method: str, size: int, overlap) -> Tuple[List[Chunk], Dict[str, str]]:
    """Chunk an entire corpus.

    Returns ``(chunks, chunk_to_doc)`` where each chunk is
    ``{"_id", "original_id", "text"}`` and ``chunk_to_doc`` maps chunk id ->
    parent document id.
    """
    chunks: List[Chunk] = []
    chunk_to_doc: Dict[str, str] = {}
    for doc in corpus:
        doc_id = doc["_id"]
        pieces = chunk_document(doc, method, size, overlap)
        if not pieces:  # keep at least the (possibly empty) doc so ids survive
            pieces = [doc_text(doc)]
        for i, piece in enumerate(pieces):
            cid = f"{doc_id}_chunk_{i}"
            chunks.append({"_id": cid, "original_id": doc_id, "text": piece})
            chunk_to_doc[cid] = doc_id
    return chunks, chunk_to_doc


def chunks_to_doc_map(chunks: List[Chunk]) -> Dict[str, str]:
    """Build chunk_id -> original_id map from pre-chunked data."""
    return {c["_id"]: c.get("original_id", c["_id"]) for c in chunks}
