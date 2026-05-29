"""Hybrid dense + BM25 retriever with on-disk index caching.

Consolidates the three slightly-different ``hybrid_search`` implementations from
the notebooks into one. Heavy deps (sentence-transformers, faiss, rank-bm25,
numpy, torch) are imported lazily so importing this module is cheap.

Typical use::

    r = HybridRetriever(cfg)
    r.build(chunks)            # chunks: [{"_id","original_id","text"}, ...]
    r.save(cache_dir)
    hits = r.retrieve("revenue of Apple", top_k_retrieval=100, alpha=0.6, top_k_docs=10)

``retrieve`` returns document-level :class:`DocHit` candidates (chunk scores
aggregated by max), ready for reranking / generation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX

_INDEX_FILE = "dense.faiss"
_BM25_FILE = "bm25.pkl"
_META_FILE = "meta.json"


@dataclass
class DocHit:
    """A document-level retrieval candidate."""

    doc_id: str
    score: float
    chunk_ids: List[str] = field(default_factory=list)
    text: str = ""


# --------------------------------------------------------------------------- #
# Pure-python score fusion (unit-testable, no numpy needed)
# --------------------------------------------------------------------------- #
def _minmax(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def fuse_scores(
    dense: Dict[int, float],
    sparse: Dict[int, float],
    alpha: float,
) -> List[tuple]:
    """Combine dense + sparse score maps (index -> score).

    Both are min-max normalised over the union of candidate indices, then
    combined as ``alpha*dense + (1-alpha)*sparse``. Returns
    ``[(index, score), ...]`` sorted by score descending.
    """
    dense_n = _minmax(dense)
    sparse_n = _minmax(sparse)
    indices = set(dense_n) | set(sparse_n)
    fused = [
        (idx, alpha * dense_n.get(idx, 0.0) + (1 - alpha) * sparse_n.get(idx, 0.0))
        for idx in indices
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_e5 = cfg.is_e5
        self.use_hybrid = cfg.use_hybrid
        self._model = None
        self._index = None          # faiss index
        self._bm25 = None
        self.chunk_ids: List[str] = []
        self.chunk_to_doc: Dict[str, str] = {}
        self.chunk_texts: List[str] = []   # raw text, for BM25 + representative snippets
        self._device = None

    # ----------------------------- model ---------------------------------- #
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            import torch  # lazy

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.cfg.embedding_model, device=self._device)
        return self._model

    def _encode(self, texts: List[str], is_query: bool):
        prefix = ""
        if self.is_e5:
            prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
        prepared = [f"{prefix}{t}" for t in texts] if prefix else texts
        return self.model.encode(
            prepared,
            batch_size=self.cfg.embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 2000,
        )

    # ----------------------------- build ---------------------------------- #
    def build(self, chunks: List[dict]) -> "HybridRetriever":
        """Build dense (FAISS) and BM25 indexes from chunks."""
        import faiss  # lazy

        self.chunk_ids = [c["_id"] for c in chunks]
        self.chunk_to_doc = {c["_id"]: c.get("original_id", c["_id"]) for c in chunks}
        self.chunk_texts = [str(c.get("text", "")) for c in chunks]

        embeddings = self._encode(self.chunk_texts, is_query=False).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index

        if self.use_hybrid:
            from rank_bm25 import BM25Okapi  # lazy

            tokenized = [t.lower().split() for t in self.chunk_texts]
            self._bm25 = BM25Okapi(tokenized)
        return self

    # ----------------------------- persistence ---------------------------- #
    def save(self, path) -> None:
        import faiss  # lazy

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        if self._bm25 is not None:
            with open(path / _BM25_FILE, "wb") as f:
                pickle.dump(self._bm25, f)
        meta = {
            "embedding_model": self.cfg.embedding_model,
            "is_e5": self.is_e5,
            "use_hybrid": self.use_hybrid,
            "chunk_ids": self.chunk_ids,
            "chunk_to_doc": self.chunk_to_doc,
            "chunk_texts": self.chunk_texts,
        }
        with open(path / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path, cfg) -> "HybridRetriever":
        import faiss  # lazy

        path = Path(path)
        with open(path / _META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        r = cls(cfg)
        r.is_e5 = meta["is_e5"]
        r.use_hybrid = meta["use_hybrid"]
        r.chunk_ids = meta["chunk_ids"]
        r.chunk_to_doc = meta["chunk_to_doc"]
        r.chunk_texts = meta["chunk_texts"]
        r._index = faiss.read_index(str(path / _INDEX_FILE))
        bm25_path = path / _BM25_FILE
        if r.use_hybrid and bm25_path.exists():
            with open(bm25_path, "rb") as f:
                r._bm25 = pickle.load(f)
        return r

    @staticmethod
    def is_cached(path) -> bool:
        path = Path(path)
        return (path / _INDEX_FILE).exists() and (path / _META_FILE).exists()

    # ----------------------------- retrieve ------------------------------- #
    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = 100,
        alpha: float = 0.6,
        top_k_docs: int = 50,
    ) -> List[DocHit]:
        """Return up to ``top_k_docs`` document candidates for *query*."""
        import numpy as np  # lazy

        q_emb = self._encode([query], is_query=True)[0].astype("float32")
        n = self._index.ntotal
        depth = min(max(top_k_retrieval, top_k_docs) * 2, n)

        d_scores, d_idx = self._index.search(q_emb.reshape(1, -1), depth)
        dense = {int(i): float(s) for i, s in zip(d_idx[0], d_scores[0]) if i >= 0}

        sparse: Dict[int, float] = {}
        if self.use_hybrid and self._bm25 is not None:
            bm = self._bm25.get_scores(query.lower().split())
            top_sparse = np.argsort(bm)[::-1][:depth]
            sparse = {int(i): float(bm[i]) for i in top_sparse}

        if sparse:
            fused = fuse_scores(dense, sparse, alpha)
        else:
            fused = sorted(dense.items(), key=lambda x: x[1], reverse=True)

        # Aggregate chunk hits to document level (max score wins).
        docs: Dict[str, DocHit] = {}
        for idx, score in fused[:top_k_retrieval]:
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            cid = self.chunk_ids[idx]
            doc_id = self.chunk_to_doc.get(cid, cid)
            hit = docs.get(doc_id)
            if hit is None:
                docs[doc_id] = DocHit(
                    doc_id=doc_id,
                    score=score,
                    chunk_ids=[cid],
                    text=self.chunk_texts[idx],
                )
            else:
                hit.chunk_ids.append(cid)
                if score > hit.score:
                    hit.score = score
                    hit.text = self.chunk_texts[idx]

        ranked = sorted(docs.values(), key=lambda h: h.score, reverse=True)
        return ranked[:top_k_docs]
