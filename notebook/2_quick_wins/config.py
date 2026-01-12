"""
Configuration for Notebook 2: Quick Wins Notebook
================================================

This config file contains all configurable parameters for the quick wins pipeline.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Base data directory (2 levels up from this folder)
DATA_DIR = Path('../../data')

# Output file (local directory)
OUTPUT_FILE = Path('./submission_improved.csv')

# =============================================================================
# DATASETS
# =============================================================================

DATASETS = [
    'convfinqa', 
    'financebench', 
    'finder',
    'finqa', 
    'finqabench', 
    'multiheirtt', 
    'tatqa'
]

# =============================================================================
# MODELS
# =============================================================================

EMBEDDING_MODEL = 'intfloat/e5-small-v2'  # Best model from evaluation (NDCG@10: 0.5513)
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================

CHUNKING_CONFIG = {
    'use_chunking': True,
    'chunk_size': 512,
    'chunk_overlap': 128,
    'chunk_aggregation': 'max',
    'preserve_tables': True,
}

# =============================================================================
# HYBRID SEARCH SETTINGS
# =============================================================================

HYBRID_CONFIG = {
    'use_hybrid': True,
    'hybrid_alpha': 0.6,  # Weight for dense (1-alpha for BM25)
}

# =============================================================================
# RETRIEVAL PARAMETERS
# =============================================================================

RETRIEVAL_CONFIG = {
    'top_k_retrieval': 100,
    'top_k_rerank': 50,
    'top_k_final': 10,
    'embed_batch_size': 16,
    'max_length': 4096,
    'eval_on_qrels': True,
}

# =============================================================================
# COMBINED CONFIG (for backward compatibility)
# =============================================================================

CONFIG = {
    'data_dir': str(DATA_DIR),
    'output_file': str(OUTPUT_FILE),
    'datasets': DATASETS,
    
    # Models
    'embedding_model': EMBEDDING_MODEL,
    'reranker_model': RERANKER_MODEL,
    
    # Chunking
    **CHUNKING_CONFIG,
    
    # Hybrid
    **HYBRID_CONFIG,
    
    # Retrieval
    **RETRIEVAL_CONFIG,
}


def print_config():
    """Print configuration summary."""
    print("\n" + "="*60)
    print("⚙️ QUICK WINS CONFIGURATION")
    print("="*60)
    
    print(f"\n📂 Data Directory: {DATA_DIR}")
    print(f"📄 Output File: {OUTPUT_FILE}")
    
    print(f"\n🤖 Models:")
    print(f"   Embedding: {EMBEDDING_MODEL}")
    print(f"   Reranker: {RERANKER_MODEL}")
    
    print(f"\n✂️ Chunking:")
    for k, v in CHUNKING_CONFIG.items():
        print(f"   {k}: {v}")
    
    print(f"\n🔍 Hybrid Search:")
    for k, v in HYBRID_CONFIG.items():
        print(f"   {k}: {v}")
    
    print(f"\n🎯 Retrieval:")
    for k, v in RETRIEVAL_CONFIG.items():
        print(f"   {k}: {v}")
    
    print("="*60)


if __name__ == "__main__":
    print_config()
