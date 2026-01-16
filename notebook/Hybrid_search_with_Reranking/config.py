# ============================================================================
# 🔍 Hybrid Search & Reranking Evaluation Configuration
# ============================================================================
# This config file contains all settings for hybrid alpha and reranker evaluation
# Import this in the notebook using: from config import *

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path('../../data')
OUTPUT_DIR = Path('./output_hybrid_rerank_eval')
CHUNKED_CORPUS_DIR = Path('../../data/chunked_corpus')
CHUNKING_CONFIG_FILE = Path('../../data/chunked_corpus/best_chunking_config_per_dataset.json')

# ============================================================================
# DATASETS TO EVALUATE
# ============================================================================
DATASETS = [
    'convfinqa',
    'financebench',
    'finder',
    'finqa',
    'finqabench',
    'multiheirtt',
    'tatqa'
]

# ============================================================================
# 🔍 HYBRID ALPHA RANGE (for Notebook 1)
# ============================================================================
# Test alpha values from 0.1 to 0.9
# Alpha = weight for dense retrieval (1-alpha = weight for BM25)
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ============================================================================
# 🤖 RERANKER MODELS TO EVALUATE (for Notebook 2)
# ============================================================================
# Format: (model_name, display_name, description)
RERANKER_MODELS = [
    # BGE Rerankers (BAAI)
    ('BAAI/bge-reranker-base', 'BGE-reranker-base', 'Base BGE reranker (278M params)'),
    ('BAAI/bge-reranker-large', 'BGE-reranker-large', 'Large BGE reranker (560M params)'),
    ('BAAI/bge-reranker-v2-m3', 'BGE-reranker-v2-m3', 'Multilingual BGE v2 (568M params)'),
    
    # Cross-encoders (sentence-transformers)
    ('cross-encoder/ms-marco-MiniLM-L-6-v2', 'MiniLM-L6-cross', 'Fast cross-encoder (22M params)'),
    ('cross-encoder/ms-marco-MiniLM-L-12-v2', 'MiniLM-L12-cross', 'Better cross-encoder (33M params)'),
    
    # MonoT5 (if available - requires transformers)
    # ('castorini/monot5-base-msmarco', 'MonoT5-base', 'T5-based reranker'),
]

# ============================================================================
# EMBEDDING MODEL (fixed for evaluation)
# ============================================================================
# Use the best embedding model from model evaluation
EMBEDDING_MODEL = 'intfloat/e5-small-v2'  # Best from notebook 0

# ============================================================================
# ⚡ SPEED OPTIMIZATION SETTINGS
# ============================================================================
SAMPLE_QUERIES_PER_DATASET = 30  # Number of queries to sample (same as model eval)
MAX_CORPUS_SAMPLE_SIZE = 500     # Max corpus size after smart sampling
USE_SMART_SAMPLING = True        # Include all relevant docs
MAX_TEXT_LENGTH = 512            # Truncate long texts

# ============================================================================
# RETRIEVAL PARAMETERS
# ============================================================================
TOP_K_RETRIEVAL = 100    # Initial retrieval (before reranking)
TOP_K_RERANK = 50        # Number of candidates to send to reranker
TOP_K_FINAL = 10         # Final results per query

# ============================================================================
# BATCH SIZES
# ============================================================================
EMBED_BATCH_SIZE = 32
RERANK_BATCH_SIZE = 16

# ============================================================================
# CHUNKING SETTINGS (use pre-chunked optimal corpus)
# ============================================================================
USE_PRECHUNKED = True    # Always use pre-chunked corpus from notebook 4
CHUNK_AGGREGATION = 'max'  # max score aggregation from chunks to docs

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVAL_ON_QRELS = True
NDCG_K = 10              # Compute NDCG@10


def print_config():
    """Print configuration summary"""
    print(f"\n📊 Configuration:")
    print(f"   Datasets: {len(DATASETS)}")
    print(f"   Alpha values to test: {ALPHA_VALUES}")
    print(f"   Reranker models: {len(RERANKER_MODELS)}")
    print(f"\n📂 Data Source:")
    print(f"   Using pre-chunked corpus: {USE_PRECHUNKED}")
    print(f"   Chunked corpus dir: {CHUNKED_CORPUS_DIR}")
    print(f"   Chunking config: {CHUNKING_CONFIG_FILE}")
    print(f"\n🎯 Retrieval Settings:")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   Top-k retrieval: {TOP_K_RETRIEVAL}")
    print(f"   Top-k rerank: {TOP_K_RERANK}")
    print(f"   Top-k final: {TOP_K_FINAL}")
    print(f"\n⚡ Speed optimizations:")
    print(f"   Queries per dataset: {SAMPLE_QUERIES_PER_DATASET}")
    print(f"   Max corpus sample: {MAX_CORPUS_SAMPLE_SIZE}")
    print(f"   Smart sampling: {USE_SMART_SAMPLING}")


def print_alpha_config():
    """Print alpha evaluation specific config"""
    print(f"\n🔍 ALPHA EVALUATION CONFIG:")
    print(f"   Alpha range: {min(ALPHA_VALUES):.1f} to {max(ALPHA_VALUES):.1f}")
    print(f"   Step size: {ALPHA_VALUES[1] - ALPHA_VALUES[0]:.1f}")
    print(f"   Total tests per dataset: {len(ALPHA_VALUES)}")
    print(f"   Total tests: {len(DATASETS) * len(ALPHA_VALUES)}")


def print_reranker_config():
    """Print reranker evaluation specific config"""
    print(f"\n🤖 RERANKER EVALUATION CONFIG:")
    print(f"   Models to test:")
    for i, (model_name, display_name, desc) in enumerate(RERANKER_MODELS, 1):
        print(f"      [{i}] {display_name}: {desc}")
    print(f"   Total tests per dataset: {len(RERANKER_MODELS)}")
    print(f"   Total tests: {len(DATASETS) * len(RERANKER_MODELS)}")
