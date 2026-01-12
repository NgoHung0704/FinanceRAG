# ============================================================================
# ✂️ Chunking Evaluation Configuration
# ============================================================================
# This config file contains all settings for chunking strategy evaluation
# Import this in the notebook using: from config import *

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path('../../data')
OUTPUT_DIR = Path('../../data/chunked_corpus')  # Chunked corpus stays in data folder
LOCAL_OUTPUT_DIR = Path('./output_chunking_eval')  # Local evaluation results

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
# Dataset characteristics (from analysis)
# ============================================================================
DATASET_INFO = {
    'multiheirtt': {
        'has_tables': True,
        'table_percentage': 67,
        'avg_length': 2956,
        'type': 'hierarchical_tables',
        'description': 'Hierarchical tables with multi-level structure'
    },
    'tatqa': {
        'has_tables': True,
        'table_percentage': 100,
        'avg_length': 2433,
        'type': 'pure_tables',
        'description': 'Pure tables with numerical reasoning'
    },
    'convfinqa': {
        'has_tables': True,
        'table_percentage': 100,
        'avg_length': 4526,
        'type': 'hybrid_tables',
        'description': 'Tables + narrative text, very long documents'
    },
    'finqa': {
        'has_tables': True,
        'table_percentage': 100,
        'avg_length': 4394,
        'type': 'hybrid_tables',
        'description': 'Tables + text, long documents'
    },
    'financebench': {
        'has_tables': False,
        'table_percentage': 0,
        'avg_length': 1359,
        'type': 'pure_text',
        'description': 'Pure text, medium length, narrative'
    },
    'finder': {
        'has_tables': False,
        'table_percentage': 0,
        'avg_length': 576,
        'type': 'short_text',
        'description': 'Very short text, large corpus (13K docs)'
    },
    'finqabench': {
        'has_tables': True,
        'table_percentage': 30.4,
        'avg_length': 1709,
        'type': 'mixed',
        'description': 'Mixed content, small corpus (92 docs)'
    }
}

# ============================================================================
# 🆕 Dataset-specific chunking strategies với SEMANTIC CHUNKING
# ============================================================================
# Max 4 strategies per dataset, prioritizing semantic chunking
DATASET_CHUNKING_STRATEGIES = {
    # 🔴 MULTIHEIRTT - HIGH PRIORITY (current: 0.1948 NDCG)
    'multiheirtt': [
        ('semantic', 3000, 0.65),
        ('no_chunking', None, None),
        ('preserve_tables', 3000, 300),
        ('row_based', None, None),
    ],
    
    # 🟠 TATQA - HIGH PRIORITY (current: 0.3408 NDCG)
    'tatqa': [
        ('semantic', 2000, 0.70),
        ('no_chunking', None, None),
        ('table_linearization', 2000, 200),
        ('preserve_tables', 4096, 400),
    ],
    
    # 🟡 CONVFINQA (current: 0.6081 NDCG)
    'convfinqa': [
        ('semantic', 2000, 0.60),
        ('recursive', 1536, 200),
        ('hybrid_semantic_table', 2000, 0.65),
        ('sliding_window', 1024, 256),
    ],
    
    # 🟡 FINQA (current: 0.5591 NDCG)
    'finqa': [
        ('semantic', 2500, 0.65),
        ('preserve_tables', 2048, 200),
        ('table_context', 2500, 300),
        ('recursive', 1536, 200),
    ],
    
    # 🟢 FINANCEBENCH (current: 1.0330 NDCG)
    'financebench': [
        ('semantic', 1000, 0.70),
        ('recursive', 768, 75),
        ('recursive', 512, 50),
        ('sentence_window', 3, None),
    ],
    
    # 🟢 FINDER (current: 0.5777 NDCG)
    'finder': [
        ('semantic', 800, 0.70),
        ('recursive', 512, 50),
        ('no_chunking', None, None),
        ('fixed', 256, 25),
    ],
    
    # 🟢 FINQABENCH (current: 1.3488 NDCG)
    'finqabench': [
        ('semantic', 800, 0.70),
        ('recursive', 512, 50),
        ('no_chunking', None, None),
        ('preserve_tables', 1024, 100),
    ]
}

# ============================================================================
# ⚡ SPEED OPTIMIZATION SETTINGS
# ============================================================================
SAMPLE_QUERIES_PER_DATASET = 30  # ⚡ Reduced from 50 to 30
USE_SMART_SAMPLING = True
MAX_CORPUS_SAMPLE_SIZE = 500     # ⚡ Reduced from 3000 to 500 (sampled BEFORE chunking)

# ============================================================================
# MODELS
# ============================================================================
EMBEDDING_MODEL = 'intfloat/e5-small-v2'  # Best model from evaluation (NDCG@10: 0.5513)
SEMANTIC_CHUNKING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def print_config():
    """Print configuration summary"""
    print(f"� PATHS:")
    print(f"  📂 Data: {DATA_DIR}")
    print(f"  📤 Chunked corpus output: {OUTPUT_DIR}")
    print(f"  📊 Evaluation results: {LOCAL_OUTPUT_DIR}")
    
    print(f"\n�📊 Will evaluate {len(DATASETS)} datasets")
    print(f"\n📋 Chunking strategies per dataset:")
    for dataset, strategies in DATASET_CHUNKING_STRATEGIES.items():
        methods = [s[0] for s in strategies]
        print(f"  {dataset}: {methods}")

    print(f"\n⚡ SPEED OPTIMIZATIONS (v2 - MUCH FASTER):")
    print(f"  📝 Queries per dataset: {SAMPLE_QUERIES_PER_DATASET}")
    print(f"  📦 Corpus sample: {MAX_CORPUS_SAMPLE_SIZE} docs (sampled BEFORE chunking!)")
    print(f"  🧠 Semantic chunking: Batch encoding (10x faster)")
    print(f"  🔢 NDCG: Fixed to always be in [0, 1]")
    print(f"  🤖 Embedding: {EMBEDDING_MODEL}")
    print(f"\n⏱️ Expected runtime: ~5-10 minutes total (vs 441+ min before!)")
