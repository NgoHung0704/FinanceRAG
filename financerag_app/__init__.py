"""FinanceRAG application package.

A clean, unified Retrieve -> Rerank -> Generate RAG stack for financial
documents, consolidated from the project's experimental notebooks.

Public entry point is :class:`financerag_app.pipeline.RAGPipeline`.
Heavy ML dependencies (torch, sentence-transformers, faiss, FlagEmbedding,
openai) are imported lazily inside the modules that need them so that the
light-weight pieces (config, data, chunking, evaluation) can be used and
tested without the full stack installed.
"""

from .config import DATASETS, AppConfig, get_dataset_config

__all__ = ["DATASETS", "AppConfig", "get_dataset_config"]
