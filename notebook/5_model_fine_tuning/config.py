# ============================================================================
# 🎯 Embedding Fine-Tuning Configuration (OPTIMIZED)
# ============================================================================
# This config file contains all settings for fine-tuning embedding models
# 
# IMPROVEMENTS based on analysis from notebooks 0, 6, 7:
# 1. E5-small is best base model (NDCG@10: 0.5513)
# 2. Train only 1 layer due to limited data (~2500 pairs)
# 3. Focus on MULTIHEIRTT query types (multi_hop, calculation)
# 4. Add E5 prefix handling (query:, passage:)
#
# Import this in the notebook using: from config import *

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path('../../data')
OUTPUT_DIR = Path('./output_finetuning')
MODELS_DIR = Path('../../models')

# ============================================================================
# DATASETS - Ordered by difficulty (hardest first for weighted sampling)
# ============================================================================
DATASETS = [
    'multiheirtt',   # HARDEST: 50.3% zero-score queries
    'finqa',
    'tatqa',
    'convfinqa',
    'finder',
    'financebench',
    'finqabench'
]

# Dataset weights for sampling (focus on hard datasets)
DATASET_WEIGHTS = {
    'multiheirtt': 2.0,   # Duple weight - worst performance
    'finqa': 1.0,
    'tatqa': 1.0,
    'convfinqa': 1.0,
    'finder': 1.0,
    'financebench': 1.0,
    'finqabench': 1.0
}

# Qrels file mapping
QRELS_MAPPING = {
    'convfinqa': 'ConvFinQA_qrels.tsv',
    'financebench': 'FinanceBench_qrels.tsv',
    'finder': 'FinDER_qrels.tsv',
    'finqa': 'FinQA_qrels.tsv',
    'finqabench': 'FinQABench_qrels.tsv',
    'multiheirtt': 'MultiHeirtt_qrels.tsv',
    'tatqa': 'TATQA_qrels.tsv',
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# E5-small is best from evaluation (NDCG@10: 0.5513)
BASE_MODEL = 'intfloat/e5-small-v2'
MINING_MODEL = 'intfloat/e5-small-v2'

# E5 models require special prefixes
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ============================================================================
# LAYER FREEZING CONFIGURATION (CRITICAL for small datasets)
# ============================================================================
# E5-small has 12 encoder layers (0-11)
# With ~2500 training pairs, train ONLY 1-2 layers to prevent overfitting
LAYER_FREEZING_CONFIG = {
    'enabled': True,
    'layers_to_train': 3,          # Train only last 3 layers (layers 9, 10, 11)
    'freeze_embeddings': True,      # Freeze word/position embeddings
    'freeze_pooler': False,         # Keep pooler trainable
    
    # Alternative: Use LoRA-like approach (train small adapter)
    'use_adapter': False,           # If True, add small trainable adapter
    'adapter_dim': 64,              # Adapter bottleneck dimension
}

# ============================================================================
# TRAINING PARAMETERS (OPTIMIZED for small data)
# ============================================================================
TRAINING_CONFIG = {
    # Data split
    'train_ratio': 0.9,
    'val_ratio': 0.1,
    
    # Training hyperparameters - CONSERVATIVE for small data
    'epochs': 20,                   # Reduced from 20
    'batch_size': 16,               # Smaller batch for more gradient updates
    'gradient_accumulation': 4,     # Effective batch = 64
    'learning_rate': 1e-5,          # Lower LR for stability
    'warmup_ratio': 0.1,
    
    # Regularization - INCREASED for small data
    'weight_decay': 0.05,           # Higher regularization
    'dropout': 0.1,                 # Add dropout
    'label_smoothing': 0.1,         # Soft labels
    
    # Text processing
    'max_seq_length': 256,          # Reduced for E5-small
    'max_doc_length': 1024,         # Shorter docs
    
    # Optimizer
    'adam_epsilon': 1e-8,
    
    # Scheduler
    'scheduler': 'cosine',          # Cosine decay
    
    # Early stopping - AGGRESSIVE
    'early_stopping_patience': 2,   # Stop early if no improvement
    'min_delta': 0.001,             # Minimum improvement threshold
    
    # Evaluation
    'evaluation_steps': 50,
    'save_best_model': True,
    
    # Random seed for reproducibility
    'seed': 42,
}

# ============================================================================
# HARD NEGATIVES MINING (FOCUSED on difficult queries)
# ============================================================================
HARD_NEGATIVES_CONFIG = {
    'top_k': 30,               # Reduced for efficiency
    'num_hard_negs': 2,        # Fewer but higher quality
    'batch_size': 32,
    
    # Mine from difficult query types specifically
    'focus_on_multiheirtt': True,
    'mine_by_query_type': True,
    
    # Query type weights for mining (from notebook 7 analysis)
    'query_type_weights': {
        'multi_hop': 3,        # Hardest: "in the year with the most..."
        'calculation': 3,      # "percentage change", "growth rate"
        'aggregation': 2,      # "sum of X and Y"
        'simple': 1,
        'table_lookup': 1      # Easiest: specific table terms
    }
}

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
LOSS_FUNCTION = 'TripletLoss'
TRIPLET_MARGIN = 0.3           # Reduced margin for small data

# ============================================================================
# QUERY AUGMENTATION (from notebook 7 solutions)
# ============================================================================
QUERY_AUGMENTATION_CONFIG = {
    'enabled': True,
    'augment_fraction': 0.3,    # Augment 30% of difficult queries
    
    # Patterns to augment (highest failure rate)
    'patterns_to_augment': [
        'multi_hop',      # "in the year with the most X, what is Y"
        'calculation',    # "percentage change between X and Y"
        'aggregation'     # "sum of X and Y"
    ],
    
    # Augmentation strategies
    'strategies': {
        'paraphrase': True,     # Rephrase the query
        'decompose': True,      # Break into sub-queries
        'expand_years': True,   # Add adjacent year context
        'add_synonyms': True    # Financial term synonyms
    }
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_MODEL_NAME = 'e5-small-financerag-finetuned-v3'


def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("📊 Fine-Tuning Configuration (OPTIMIZED)")
    print("=" * 70)
    
    print(f"\n🤖 Models:")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Mining model: {MINING_MODEL}")
    print(f"   E5 prefixes: query='{E5_QUERY_PREFIX}', passage='{E5_PASSAGE_PREFIX}'")
    
    print(f"\n🧊 Layer Freezing:")
    for key, value in LAYER_FREEZING_CONFIG.items():
        print(f"   {key}: {value}")
    
    print(f"\n📁 Datasets ({len(DATASETS)}):")
    for ds in DATASETS:
        weight = DATASET_WEIGHTS.get(ds, 1.0)
        marker = "⭐" if weight > 1 else ""
        print(f"   - {ds} (weight: {weight}) {marker}")
    
    print(f"\n⚙️ Training Parameters:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 Hard Negatives Mining:")
    for key, value in HARD_NEGATIVES_CONFIG.items():
        if key != 'query_type_weights':
            print(f"   {key}: {value}")
    
    print(f"\n📝 Query Augmentation:")
    print(f"   enabled: {QUERY_AUGMENTATION_CONFIG['enabled']}")
    print(f"   patterns: {QUERY_AUGMENTATION_CONFIG['patterns_to_augment']}")
    
    print(f"\n📤 Output: {OUTPUT_MODEL_NAME}")
    print("=" * 70)
