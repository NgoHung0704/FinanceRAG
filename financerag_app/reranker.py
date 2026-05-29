"""Cross-encoder reranking (BAAI/bge-reranker-v2-m3 via FlagEmbedding).

Rescoring the top retrieval candidates with a cross-encoder was the single
biggest quality lever in the notebooks. This wraps it behind a small interface
and degrades gracefully (keeps retrieval order) if FlagEmbedding is missing.
"""

from __future__ import annotations

from typing import List

from .retriever import DocHit


class CrossEncoderReranker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self.available = True

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker  # lazy
            import torch  # lazy

            self._model = FlagReranker(
                self.cfg.reranker_model,
                use_fp16=torch.cuda.is_available(),
            )
        return self._model

    def rerank(self, query: str, hits: List[DocHit], top_k: int) -> List[DocHit]:
        """Re-score ``hits`` against *query* and return the top_k by new score."""
        if not hits:
            return []
        try:
            pairs = [[query, h.text[: self.cfg.max_passage_chars]] for h in hits]
            scores = self.model.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
        except Exception:
            # FlagEmbedding unavailable or failed -> keep retrieval order.
            self.available = False
            return hits[:top_k]

        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
