# ============================================================================
# 🚀 Improved Chunking Pipeline Configuration
# ============================================================================
# This config file contains all settings for the improved chunking pipeline
# Import this in the notebook using: from config import CONFIG, DATASET_SPECIFIC_CONFIG

# ============================================================================
# MAIN CONFIGURATION
# ============================================================================
CONFIG = {
    'data_dir': '../../data',
    'output_file': './submission_optimal_chunking.csv',
    
    'datasets': [
        'convfinqa', 'financebench', 'finder',
        'finqa', 'finqabench', 'multiheirtt', 'tatqa'
    ],
    
    # Models - Best from evaluation
    'embedding_model': 'intfloat/e5-small-v2',  # Best model (NDCG@10: 0.5513)
    'reranker_model': 'BAAI/bge-reranker-v2-m3',
    
    # Chunking - OPTIMAL (from notebook 3 evaluation)
    'use_prechunked': True,  # Load pre-chunked data from notebook 3
    'chunked_corpus_dir': '../../data/chunked_corpus',  # Pre-chunked files location
    'chunking_config_file': '../../data/chunked_corpus/best_chunking_config_per_dataset.json',  # Dataset-specific configs
    'use_chunking': True,
    'chunking_method': 'fixed',  # Fallback if pre-chunked not available
    'chunk_size': 512,  # characters
    'chunk_overlap': 50,  # characters
    'chunk_aggregation': 'max',  # max score aggregation
    'preserve_tables': True,  # Keep tables intact
    
    # Hybrid Search
    'use_hybrid': True,
    'hybrid_alpha': 0.6,  # 60% dense, 40% BM25
    
    # Retrieval Parameters
    'top_k_retrieval': 100,  # Initial retrieval
    'top_k_rerank': 50,  # Send to reranker
    'top_k_final': 10,  # Final results per query
    
    # Batch Sizes
    'embed_batch_size': 16,
    'rerank_batch_size': 16,
    'max_length': 512,
    
    # Evaluation
    'eval_on_qrels': True,
}

# ============================================================================
# 🎯 DATASET-SPECIFIC OVERRIDES
# ============================================================================
# MultiHeirTT và TATQA cần cấu hình đặc biệt vì queries yêu cầu numerical reasoning
DATASET_SPECIFIC_CONFIG = {
    'multiheirtt': {
        'top_k_retrieval': 200,    # Tăng vì cần retrieve 4-16 docs per query
        'top_k_rerank': 80,        # Rerank nhiều hơn
        'hybrid_alpha': 0.4,       # Tăng weight cho BM25 (60% BM25, 40% Dense)
                                   # BM25 tốt hơn cho exact number matching
    },
    'tatqa': {
        'top_k_retrieval': 150,    # Tables cũng cần numerical matching
        'top_k_rerank': 60,
        'hybrid_alpha': 0.5,       # 50-50 balance
    },
    'finqa': {
        'top_k_retrieval': 120,
        'hybrid_alpha': 0.55,
    },
    'convfinqa': {
        'top_k_retrieval': 120,
        'hybrid_alpha': 0.55,
    },
}


def print_config():
    """Print configuration summary"""
    print("✅ Configuration loaded")
    print("\n🎯 CHUNKING STRATEGY:")
    if CONFIG['use_prechunked']:
        print(f"   ✅ Using PRE-CHUNKED corpus from notebook 2")
        print(f"   📂 Source: {CONFIG['chunked_corpus_dir']}")
        print(f"   📋 Config: {CONFIG['chunking_config_file']}")
        print(f"   ℹ️ Each dataset uses its OPTIMAL chunking method")
    else:
        print(f"   Method: {CONFIG['chunking_method']}")
        print(f"   Chunk Size: {CONFIG['chunk_size']} characters")
        print(f"   Overlap: {CONFIG['chunk_overlap']} characters")
        print(f"   Table Preservation: {CONFIG['preserve_tables']}")
    print(f"\n🔗 Aggregation: {CONFIG['chunk_aggregation']} (chunks → docs)")

    print("\n🔧 DATASET-SPECIFIC OVERRIDES:")
    for ds, overrides in DATASET_SPECIFIC_CONFIG.items():
        print(f"   {ds}: {overrides}")
