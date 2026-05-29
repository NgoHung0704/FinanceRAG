"""Optional LightRAG (graph-RAG) retriever, for empirical comparison only.

This wraps `lightrag-hku <https://github.com/HKUDS/LightRAG>`_ so it can be put
through the *same* NDCG@10 harness as the hybrid retriever. LightRAG is built
for generative QA, not ranked retrieval, so we:

1. ingest each corpus document tagged with a parseable ``DOCID::<id>::`` marker;
2. query in **retrieve-only** mode (``only_need_context=True``) — no answer
   generation, no extra LLM cost at query time;
3. parse the ``DOCID`` markers out of the returned context, in order of first
   appearance, to produce a ranked document list for NDCG.

⚠️ Caveats (read before trusting numbers):
* Needs ``pip install lightrag-hku`` and an LLM key (the LLM builds the knowledge
  graph at ingest time — the expensive step). Configurable via env:
  - LLM: defaults to OpenAI ``gpt-4o-mini``; point it at a cheap OpenAI-compatible
    provider like DeepSeek with LIGHTRAG_LLM_BASE_URL / LIGHTRAG_LLM_MODEL /
    DEEPSEEK_API_KEY (see .env.example).
  - Embeddings: default is a FREE local model (``BAAI/bge-small-en-v1.5``); note
    DeepSeek has no embedding endpoint. Set LIGHTRAG_EMBED=openai to use OpenAI.
* The embedding model differs from the hybrid baseline's e5, so a win/loss mixes
  "graph vs no-graph" with "different embeddings" — read it as "does LightRAG-as-
  configured do better", not a clean graph ablation.
* DOCID markers survive only in chunks that contain a document's start, so
  recovery is most reliable on short-document datasets (finqabench, financebench).
* LightRAG's API has shifted across releases; imports/init below are defensive
  but may need a small tweak for your installed version.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import List

from .data import Document, doc_text

_DOCID = re.compile(r"DOCID::(.*?)::")


def _marker(doc_id: str) -> str:
    return f"DOCID::{doc_id}::"


def parse_doc_ids(context: str, top_k: int) -> List[str]:
    """Extract ranked, de-duplicated doc ids from a LightRAG context string.

    Order = first appearance of each ``DOCID::<id>::`` marker (≈ relevance order
    of the retrieved sources). Pure function, unit-tested.
    """
    seen, ranked = set(), []
    for m in _DOCID.findall(context or ""):
        if m not in seen:
            seen.add(m)
            ranked.append(m)
        if len(ranked) >= top_k:
            break
    return ranked


def _run(coro):
    """Run an async coroutine from sync code (handles 'no running loop')."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _load_classes():
    """Import LightRAG core classes, tolerating version differences."""
    from lightrag import LightRAG, QueryParam  # noqa: WPS433

    try:
        from lightrag.utils import EmbeddingFunc  # type: ignore
    except Exception:  # pragma: no cover
        from lightrag import EmbeddingFunc  # type: ignore
    return LightRAG, QueryParam, EmbeddingFunc


def _openai_complete():
    """The OpenAI-compatible completion helper (works for DeepSeek too)."""
    try:
        from lightrag.llm.openai import openai_complete_if_cache  # type: ignore
    except Exception:  # pragma: no cover - older layout
        from lightrag.llm import openai_complete_if_cache  # type: ignore
    return openai_complete_if_cache


def _build_llm_func():
    """LLM used to extract the knowledge graph at ingest time.

    Configurable so you can point it at a cheap OpenAI-compatible provider like
    DeepSeek (set LIGHTRAG_LLM_BASE_URL=https://api.deepseek.com/v1,
    LIGHTRAG_LLM_MODEL=deepseek-chat, DEEPSEEK_API_KEY=...). This is the
    expensive part of LightRAG, so a cheap LLM here matters most.
    """
    complete = _openai_complete()
    base_url = os.getenv("LIGHTRAG_LLM_BASE_URL") or None
    model = os.getenv("LIGHTRAG_LLM_MODEL", "gpt-4o-mini")
    api_key = (
        os.getenv("LIGHTRAG_LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await complete(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    return llm_func


def _build_embedding_func(EmbeddingFunc):
    """Embeddings for LightRAG's vector layer.

    Default = a FREE local sentence-transformers model (no embedding API cost;
    note DeepSeek has no embedding endpoint). Set LIGHTRAG_EMBED=openai to use
    OpenAI text-embedding-3-small instead.
    """
    kind = os.getenv("LIGHTRAG_EMBED", "local").lower()

    if kind == "openai":
        try:
            from lightrag.llm.openai import openai_embed  # type: ignore
        except Exception:  # pragma: no cover
            from lightrag.llm import openai_embed  # type: ignore
        model = os.getenv("LIGHTRAG_EMBED_MODEL", "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")

        async def ef(texts):
            return await openai_embed(texts, model=model, api_key=api_key)

        return EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=ef)

    # local (free) — bge-small needs no query/passage prefixes, unlike e5
    from sentence_transformers import SentenceTransformer  # lazy

    name = os.getenv("LIGHTRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    st_model = SentenceTransformer(name)
    dim = st_model.get_sentence_embedding_dimension()

    async def ef(texts):
        return st_model.encode(list(texts), normalize_embeddings=True)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=512, func=ef)


class LightRAGRetriever:
    def __init__(self, cfg, dataset: str, mode: str = "hybrid"):
        self.cfg = cfg
        self.dataset = dataset
        self.mode = mode  # naive | local | global | hybrid | mix
        self.working_dir = Path(cfg.index_cache_dir) / f"lightrag_{dataset}"
        self._rag = None
        self._QueryParam = None

    # ------------------------------------------------------------------ #
    def _get_rag(self):
        if self._rag is not None:
            return self._rag
        LightRAG, QueryParam, EmbeddingFunc = _load_classes()
        self._QueryParam = QueryParam
        self.working_dir.mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=_build_llm_func(),
            embedding_func=_build_embedding_func(EmbeddingFunc),
        )
        # Newer LightRAG requires explicit async storage init.
        for attr in ("initialize_storages",):
            init = getattr(rag, attr, None)
            if init is not None:
                maybe = init()
                if asyncio.iscoroutine(maybe):
                    _run(maybe)
        try:  # pipeline status (newer)
            from lightrag.kg.shared_storage import initialize_pipeline_status  # type: ignore

            maybe = initialize_pipeline_status()
            if asyncio.iscoroutine(maybe):
                _run(maybe)
        except Exception:
            pass

        self._rag = rag
        return rag

    # ------------------------------------------------------------------ #
    def build(self, corpus: List[Document]) -> "LightRAGRetriever":
        """Ingest the corpus into LightRAG (builds the knowledge graph).

        This is the expensive, LLM-heavy step. LightRAG persists to
        ``working_dir``, so a second run with the same dir reuses the graph.
        """
        rag = self._get_rag()
        docs = [f"{_marker(d['_id'])}\n{doc_text(d)}" for d in corpus if doc_text(d).strip()]
        # Prefer a single batched insert; fall back to per-doc.
        try:
            rag.insert(docs)
        except TypeError:
            for d in docs:
                rag.insert(d)
        return self

    @staticmethod
    def is_built(cfg, dataset: str) -> bool:
        wd = Path(cfg.index_cache_dir) / f"lightrag_{dataset}"
        # LightRAG writes several kv/graph json/graphml files into working_dir.
        return wd.exists() and any(wd.iterdir())

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Return ranked doc ids parsed from LightRAG's retrieved context."""
        rag = self._get_rag()
        param = self._QueryParam(mode=self.mode, only_need_context=True, top_k=max(top_k, 40))
        context = rag.query(query, param=param)
        if asyncio.iscoroutine(context):
            context = _run(context)
        if not isinstance(context, str):
            context = str(context)
        return parse_doc_ids(context, top_k)
