"""End-to-end RAG pipeline: retrieve -> rerank -> generate.

This is the single public entry point that the Streamlit app and the scripts
use. Per-dataset indexes are built once and cached on disk (and in memory), so
interactive queries are fast after the first build.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import data as data_mod
from .chunking import chunk_corpus, chunks_to_doc_map
from .config import AppConfig, get_dataset_config
from .generator import GenerationResult, OpenAIGenerator, Passage
from .reranker import CrossEncoderReranker
from .retriever import DocHit, HybridRetriever


@dataclass
class RetrievedPassage:
    rank: int
    doc_id: str
    title: str
    score: float
    snippet: str       # best matching chunk (for display)
    text: str          # fuller doc text (fed to LLM)


@dataclass
class QueryResult:
    question: str
    dataset: str
    passages: List[RetrievedPassage] = field(default_factory=list)
    answer: str = ""
    generation: Optional[GenerationResult] = None
    timings: Dict[str, float] = field(default_factory=dict)


class _DatasetIndex:
    """Holds a built retriever + corpus lookup for one dataset."""

    def __init__(self, retriever: HybridRetriever, corpus_lookup: Dict[str, dict]):
        self.retriever = retriever
        self.corpus_lookup = corpus_lookup


class RAGPipeline:
    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or AppConfig()
        self.reranker = CrossEncoderReranker(self.cfg)
        self.generator = OpenAIGenerator(self.cfg)
        self._indexes: Dict[str, _DatasetIndex] = {}

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #
    def _cache_dir(self, dataset: str) -> Path:
        model_tag = Path(self.cfg.embedding_model).name.replace("/", "_")
        return self.cfg.index_cache_dir / f"{dataset}__{model_tag}"

    def _prepare_chunks(self, dataset: str) -> List[dict]:
        """Load pre-chunked corpus if available, else chunk on the fly."""
        if self.cfg.use_prechunked:
            chunks = data_mod.load_prechunked(dataset, self.cfg.chunked_corpus_dir)
            if chunks:
                return chunks
        corpus = data_mod.load_corpus(dataset, self.cfg.data_dir)
        dc = get_dataset_config(dataset)
        chunks, _ = chunk_corpus(corpus, dc.chunk_method, dc.chunk_size, dc.chunk_overlap)
        return chunks

    def build_index(self, dataset: str, force: bool = False) -> None:
        """Build (and cache to disk) the retriever index for a dataset."""
        cache = self._cache_dir(dataset)
        if not force and HybridRetriever.is_cached(cache):
            return
        chunks = self._prepare_chunks(dataset)
        retriever = HybridRetriever(self.cfg).build(chunks)
        retriever.save(cache)

    def load_dataset(self, dataset: str, rebuild: bool = False) -> _DatasetIndex:
        """Return an in-memory index for a dataset, building/loading as needed."""
        if not rebuild and dataset in self._indexes:
            return self._indexes[dataset]

        cache = self._cache_dir(dataset)
        if rebuild or not HybridRetriever.is_cached(cache):
            self.build_index(dataset, force=rebuild)
        retriever = HybridRetriever.load(cache, self.cfg)

        corpus = data_mod.load_corpus(dataset, self.cfg.data_dir)
        lookup = data_mod.build_corpus_lookup(corpus)

        idx = _DatasetIndex(retriever, lookup)
        self._indexes[dataset] = idx
        return idx

    def index_ready(self, dataset: str) -> bool:
        return dataset in self._indexes or HybridRetriever.is_cached(self._cache_dir(dataset))

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def _retrieve_and_rerank(
        self,
        dataset: str,
        question: str,
        top_k: int,
        alpha: Optional[float],
    ) -> List[RetrievedPassage]:
        dc = get_dataset_config(dataset)
        idx = self.load_dataset(dataset)
        use_alpha = dc.hybrid_alpha if alpha is None else alpha

        hits: List[DocHit] = idx.retriever.retrieve(
            question,
            top_k_retrieval=dc.top_k_retrieval,
            alpha=use_alpha,
            top_k_docs=dc.top_k_rerank,
        )

        # Keep the best-chunk snippet, then swap in fuller doc text for reranking.
        snippets = {h.doc_id: h.text for h in hits}
        for h in hits:
            doc = idx.corpus_lookup.get(h.doc_id)
            if doc:
                h.text = data_mod.doc_text(doc)[: self.cfg.max_passage_chars]

        if self.cfg.use_reranker:
            hits = self.reranker.rerank(question, hits, top_k)
        else:
            hits = hits[:top_k]

        passages: List[RetrievedPassage] = []
        for rank, h in enumerate(hits, start=1):
            doc = idx.corpus_lookup.get(h.doc_id, {})
            passages.append(
                RetrievedPassage(
                    rank=rank,
                    doc_id=h.doc_id,
                    title=str(doc.get("title", "")),
                    score=h.score,
                    snippet=snippets.get(h.doc_id, h.text)[:600],
                    text=h.text,
                )
            )
        return passages

    def query(
        self,
        dataset: str,
        question: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        generate: Optional[bool] = None,
    ) -> QueryResult:
        """Run the full pipeline for one question."""
        top_k = top_k or self.cfg.top_k_final
        do_generate = self.cfg.use_generation if generate is None else generate
        timings: Dict[str, float] = {}

        t0 = time.time()
        passages = self._retrieve_and_rerank(dataset, question, top_k, alpha)
        timings["retrieve_rerank_s"] = round(time.time() - t0, 3)

        result = QueryResult(question=question, dataset=dataset, passages=passages, timings=timings)

        if do_generate:
            t1 = time.time()
            gen = self.generator.generate(
                question,
                [Passage(doc_id=p.doc_id, text=p.text, title=p.title) for p in passages],
            )
            timings["generation_s"] = round(time.time() - t1, 3)
            result.generation = gen
            result.answer = gen.answer
        return result

    # ------------------------------------------------------------------ #
    # Batch retrieval (for NDCG evaluation / submission parity)
    # ------------------------------------------------------------------ #
    def batch_retrieve(
        self,
        dataset: str,
        questions: List[dict],
        top_k: int = 10,
    ) -> Dict[str, List[str]]:
        """Return ``{query_id: [doc_id, ...]}`` for a list of query dicts."""
        results: Dict[str, List[str]] = {}
        for q in questions:
            passages = self._retrieve_and_rerank(dataset, q["text"], top_k, alpha=None)
            results[q["_id"]] = [p.doc_id for p in passages]
        return results
