# ============================================================================
# 📊 Base Notebook Configuration - FinanceRAG Pipeline
# ============================================================================
# This config file contains all settings for the base retrieval pipeline
# Import this in the notebook using: from config import CONFIG

# Pipeline Configuration
CONFIG = {
    # Paths
    'data_dir': '../../data',
    'output_file': './submission.csv',
    
    # Datasets to process (7 financial datasets)
    'datasets': [
        'convfinqa',
        'financebench', 
        'finder',
        'finqa',
        'finqabench',
        'multiheirtt',
        'tatqa'
    ],
    
    # Model names
    'embedding_model': 'intfloat/e5-small-v2',  # Best model from evaluation (NDCG@10: 0.5513)
    'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Fast and effective
    
    # Retrieval parameters
    'top_k_retrieval': 50,  # Retrieve top-50 candidates
    'top_k_final': 10,       # Rerank to top-10 for submission
    
    # Batch sizes (reduced to prevent OOM)
    'embed_batch_size': 8,   # Reduced from 32 - financial docs are very long
    'rerank_batch_size': 8,  # Reduced from 16
    
    # Text length limit (to prevent OOM)
    'max_length': 4096,      # Truncate very long documents (BGE-M3 supports 8192 but uses too much memory)
}


def print_config():
    """Print configuration summary"""
    print("Configuration loaded:")
    for key, value in CONFIG.items():
        if key != 'datasets':
            print(f"  {key}: {value}")
    print(f"  datasets: {len(CONFIG['datasets'])} datasets")
