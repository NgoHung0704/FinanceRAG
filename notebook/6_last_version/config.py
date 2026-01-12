# ============================================================================
# 🏆 FINAL SUBMISSION CONFIGURATION
# ============================================================================
# Combines:
# 1. Fine-tuned E5-small model (from notebook 5)
# 2. Optimal semantic chunking (from notebook 3/4)
# 3. Dataset-specific hybrid search (from notebook 4)
# 4. Best practices from all notebooks

from pathlib import Path
import os

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path('../../data')
MODELS_DIR = Path('../../models')
OUTPUT_DIR = Path('./output')
CHUNKED_CORPUS_DIR = DATA_DIR / 'chunked_corpus'

# ============================================================================
# DATASETS
# ============================================================================
DATASETS = [
    'convfinqa', 'financebench', 'finder',
    'finqa', 'finqabench', 'multiheirtt', 'tatqa'
]

# ============================================================================
# MODELS - FINE-TUNED + BEST RERANKER
# ============================================================================
# Fine-tuned E5-small on FinanceRAG data
FINETUNED_MODEL = MODELS_DIR / 'e5-small-financerag-finetuned-v3'

# Check if fine-tuned model exists
if FINETUNED_MODEL.exists():
    EMBEDDING_MODEL = str(FINETUNED_MODEL)
    USE_FINETUNED = True
else:
    EMBEDDING_MODEL = 'intfloat/e5-small-v2'
    USE_FINETUNED = False

# Fallback base model
BASE_MODEL = 'intfloat/e5-small-v2'

# Best reranker from evaluation
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

# E5 prefixes (REQUIRED for E5 models!)
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ============================================================================
# MAIN CONFIGURATION
# ============================================================================
CONFIG = {
    # Paths
    'data_dir': str(DATA_DIR),
    'output_file': './submission_final.csv',
    'chunked_corpus_dir': str(CHUNKED_CORPUS_DIR),
    'chunking_config_file': str(CHUNKED_CORPUS_DIR / 'best_chunking_config_per_dataset.json'),
    
    'datasets': DATASETS,
    
    # Models
    'embedding_model': EMBEDDING_MODEL,
    'base_model': BASE_MODEL,
    'reranker_model': RERANKER_MODEL,
    'use_finetuned': USE_FINETUNED,
    
    # E5 prefixes
    'e5_query_prefix': E5_QUERY_PREFIX,
    'e5_passage_prefix': E5_PASSAGE_PREFIX,
    
    # Chunking - Use pre-chunked data from notebook 3
    'use_prechunked': True,
    'use_chunking': True,
    'chunking_method': 'semantic',  # Best from evaluation
    'chunk_size': 512,
    'chunk_overlap': 50,
    'chunk_aggregation': 'max',
    'preserve_tables': True,
    
    # Hybrid Search
    'use_hybrid': True,
    'hybrid_alpha': 0.6,  # 60% dense, 40% BM25
    
    # Retrieval Parameters
    'top_k_retrieval': 100,
    'top_k_rerank': 50,
    'top_k_final': 10,
    
    # Batch Sizes (optimized for GPU memory)
    'embed_batch_size': 32,
    'rerank_batch_size': 16,
    'max_length': 512,
    
    # Evaluation
    'eval_on_qrels': True,
}

# ============================================================================
# 🎯 DATASET-SPECIFIC OVERRIDES
# ============================================================================
# Based on analysis from notebooks 6 & 7 (MULTIHEIRTT analysis)
DATASET_SPECIFIC_CONFIG = {
    'multiheirtt': {
        'top_k_retrieval': 200,    # Cần retrieve nhiều vì multi-doc queries
        'top_k_rerank': 80,
        'hybrid_alpha': 0.4,       # 60% BM25 - tốt cho numerical matching
    },
    'tatqa': {
        'top_k_retrieval': 150,
        'top_k_rerank': 60,
        'hybrid_alpha': 0.5,       # 50-50 balanced
    },
    'finqa': {
        'top_k_retrieval': 120,
        'hybrid_alpha': 0.55,
    },
    'convfinqa': {
        'top_k_retrieval': 120,
        'hybrid_alpha': 0.55,
    },
    'financebench': {
        'hybrid_alpha': 0.7,       # Semantic-heavy for long documents
    },
    'finder': {
        'hybrid_alpha': 0.65,
    },
    'finqabench': {
        'hybrid_alpha': 0.6,
    },
}

# ============================================================================
# QRELS MAPPING
# ============================================================================
QRELS_MAPPING = {
    'convfinqa': 'ConvFinQA_qrels.tsv',
    'financebench': 'FinanceBench_qrels.tsv',
    'finder': 'FinDER_qrels.tsv',
    'finqa': 'FinQA_qrels.tsv',
    'finqabench': 'FinQABench_qrels.tsv',
    'multiheirtt': 'MultiHeirtt_qrels.tsv',
    'tatqa': 'TATQA_qrels.tsv',
}


def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("🏆 FINAL SUBMISSION CONFIGURATION")
    print("=" * 70)
    
    print("\n📊 MODELS:")
    if CONFIG['use_finetuned']:
        print(f"   ✅ Embedding: {CONFIG['embedding_model']} (FINE-TUNED)")
    else:
        print(f"   ⚠️ Embedding: {CONFIG['embedding_model']} (base model)")
    print(f"   Reranker: {CONFIG['reranker_model']}")
    
    print("\n✂️ CHUNKING:")
    if CONFIG['use_prechunked']:
        print(f"   ✅ Using PRE-CHUNKED corpus (semantic chunking)")
        print(f"   📂 Source: {CONFIG['chunked_corpus_dir']}")
    else:
        print(f"   Method: {CONFIG['chunking_method']}")
    print(f"   Aggregation: {CONFIG['chunk_aggregation']}")
    
    print("\n🔍 RETRIEVAL:")
    print(f"   Hybrid: {CONFIG['use_hybrid']} (alpha={CONFIG['hybrid_alpha']})")
    print(f"   Top-K: retrieval={CONFIG['top_k_retrieval']}, rerank={CONFIG['top_k_rerank']}, final={CONFIG['top_k_final']}")
    
    print("\n🎯 DATASET OVERRIDES:")
    for ds, overrides in DATASET_SPECIFIC_CONFIG.items():
        print(f"   {ds}: {overrides}")
    
    print("=" * 70)
