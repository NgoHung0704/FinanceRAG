"""Single source of truth for FinanceRAG app configuration.

Consolidates the per-dataset settings that were previously duplicated across
the experiment notebooks:

* chunking method / size / overlap  (from data/chunked_corpus/dataset_chunking_method_mapping.json)
* hybrid alpha (dense vs BM25 weight)  (from the per-dataset tuning experiments)
* retrieval depths

Everything here is plain data + small dataclasses, no heavy imports, so it is
safe to import anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repository root = parent of this package)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
CHUNKED_CORPUS_DIR = DATA_DIR / "chunked_corpus"
INDEX_CACHE_DIR = REPO_ROOT / ".rag_cache"  # built FAISS/BM25 indexes live here

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
DATASETS = [
    "convfinqa",
    "financebench",
    "finder",
    "finqa",
    "finqabench",
    "multiheirtt",
    "tatqa",
]

# Human-friendly labels + one-line descriptions for the UI.
DATASET_INFO: Dict[str, Dict[str, str]] = {
    "convfinqa": {"label": "ConvFinQA", "desc": "Conversational financial QA over tables + narrative"},
    "financebench": {"label": "FinanceBench", "desc": "Open-book financial question answering"},
    "finder": {"label": "FinDER", "desc": "Financial document retrieval (10-K excerpts)"},
    "finqa": {"label": "FinQA", "desc": "Numerical reasoning over financial reports"},
    "finqabench": {"label": "FinQABench", "desc": "Financial QA benchmark (small corpus)"},
    "multiheirtt": {"label": "MultiHierTT", "desc": "Multi-hierarchical table reasoning"},
    "tatqa": {"label": "TAT-QA", "desc": "Question answering over tables and text"},
}

QRELS_FILENAMES: Dict[str, str] = {
    "convfinqa": "ConvFinQA_qrels.tsv",
    "financebench": "FinanceBench_qrels.tsv",
    "finder": "FinDER_qrels.tsv",
    "finqa": "FinQA_qrels.tsv",
    "finqabench": "FinQABench_qrels.tsv",
    "multiheirtt": "MultiHeirtt_qrels.tsv",
    "tatqa": "TATQA_qrels.tsv",
}

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Prefer a fine-tuned E5-small under models/ if present, else the base model.
_FINETUNED = MODELS_DIR / "e5-small-financerag-finetuned-v3"
DEFAULT_EMBEDDING_MODEL = str(_FINETUNED) if _FINETUNED.exists() else "intfloat/e5-small-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # cheap + strong; override via env or UI

# E5 models require these prefixes.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


# --------------------------------------------------------------------------- #
# Per-dataset retrieval configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Retrieval/chunking knobs tuned per dataset (from the experiments)."""

    name: str
    chunk_method: str = "recursive"          # recursive | preserve_tables | semantic | none
    chunk_size: int = 512
    chunk_overlap: int = 50                   # chars (or similarity threshold for 'semantic')
    hybrid_alpha: float = 0.6                 # weight on dense; (1-alpha) on BM25
    top_k_retrieval: int = 100
    top_k_rerank: int = 50


# Values distilled from dataset_chunking_method_mapping.json (chunk_*) and the
# per-dataset tuning experiments (alpha / top_k). For 'semantic',
# chunk_overlap holds the similarity threshold.
_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "convfinqa": DatasetConfig("convfinqa", "semantic", 2000, 0.6, hybrid_alpha=0.55, top_k_retrieval=120),
    "financebench": DatasetConfig("financebench", "semantic", 1000, 0.7, hybrid_alpha=0.70, top_k_retrieval=100),
    "finder": DatasetConfig("finder", "recursive", 512, 50, hybrid_alpha=0.65, top_k_retrieval=100),
    "finqa": DatasetConfig("finqa", "semantic", 2500, 0.65, hybrid_alpha=0.55, top_k_retrieval=120),
    "finqabench": DatasetConfig("finqabench", "preserve_tables", 1024, 100, hybrid_alpha=0.60, top_k_retrieval=100),
    "multiheirtt": DatasetConfig("multiheirtt", "semantic", 3000, 0.65, hybrid_alpha=0.40, top_k_retrieval=200, top_k_rerank=80),
    "tatqa": DatasetConfig("tatqa", "semantic", 2000, 0.7, hybrid_alpha=0.50, top_k_retrieval=150, top_k_rerank=60),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Return the tuned config for *name*, or a sane default for unknown datasets."""
    key = name.lower()
    return _DATASET_CONFIGS.get(key, DatasetConfig(name=key))


# --------------------------------------------------------------------------- #
# Global app configuration
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    """Runtime configuration for the RAG pipeline.

    Defaults can be overridden via environment variables so the same code runs
    locally, in Docker, or on a server without edits.
    """

    data_dir: Path = DATA_DIR
    chunked_corpus_dir: Path = CHUNKED_CORPUS_DIR
    index_cache_dir: Path = INDEX_CACHE_DIR

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    llm_model: str = DEFAULT_LLM_MODEL

    use_hybrid: bool = True
    use_reranker: bool = True
    use_generation: bool = True

    top_k_final: int = 10            # passages shown / fed to the LLM
    embed_batch_size: int = 32
    rerank_batch_size: int = 16
    max_passage_chars: int = 2048    # truncation before rerank/generation

    # Prefer pre-chunked corpora under data/chunked_corpus/ when present.
    use_prechunked: bool = True

    openai_api_key: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        # Environment overrides.
        self.embedding_model = os.getenv("FINRAG_EMBEDDING_MODEL", self.embedding_model)
        self.reranker_model = os.getenv("FINRAG_RERANKER_MODEL", self.reranker_model)
        self.llm_model = os.getenv("FINRAG_LLM_MODEL", self.llm_model)
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")

        if os.getenv("FINRAG_DATA_DIR"):
            self.data_dir = Path(os.environ["FINRAG_DATA_DIR"])
        if os.getenv("FINRAG_CACHE_DIR"):
            self.index_cache_dir = Path(os.environ["FINRAG_CACHE_DIR"])

        # Normalise to Path objects.
        self.data_dir = Path(self.data_dir)
        self.chunked_corpus_dir = Path(self.chunked_corpus_dir)
        self.index_cache_dir = Path(self.index_cache_dir)

    @property
    def is_e5(self) -> bool:
        """Whether the embedding model needs E5 query:/passage: prefixes."""
        return "e5" in self.embedding_model.lower()
