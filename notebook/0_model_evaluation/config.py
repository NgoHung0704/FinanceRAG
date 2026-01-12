# ============================================================================
# 🔍 Embedding Model Evaluation Configuration
# ============================================================================
# This config file contains all settings for embedding model evaluation
# Import this in the notebook using: from config import *

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = Path('../../data')
OUTPUT_DIR = Path('./output_embedding_eval')

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
# 🤖 EMBEDDING MODELS TO EVALUATE
# ============================================================================
# Format: (model_name, display_name, description)
EMBEDDING_MODELS = [
    # General-purpose models
    ('sentence-transformers/all-MiniLM-L6-v2', 'MiniLM-L6', 'Fast, lightweight (22M params)'),
    ('sentence-transformers/all-mpnet-base-v2', 'MPNet-base', 'Strong general purpose (110M params)'),
    
    # BGE models (Chinese Academy of Sciences)
    ('BAAI/bge-small-en-v1.5', 'BGE-small', 'Small BGE (33M params)'),
    ('BAAI/bge-base-en-v1.5', 'BGE-base', 'Base BGE (110M params)'),
    ('BAAI/bge-large-en-v1.5', 'BGE-large', 'Large BGE (335M params)'),
    
    # E5 models (Microsoft)
    ('intfloat/e5-small-v2', 'E5-small', 'Small E5 (33M params)'),
    ('intfloat/e5-base-v2', 'E5-base', 'Base E5 (110M params)'),
    ('intfloat/e5-large-v2', 'E5-large', 'Large E5 (335M params)'),
    
    # GTE models (Alibaba)
    ('thenlper/gte-small', 'GTE-small', 'Small GTE (33M params)'),
    ('thenlper/gte-base', 'GTE-base', 'Base GTE (110M params)'),
    ('thenlper/gte-large', 'GTE-large', 'Large GTE (335M params)'),
    
    # Finance-specific (if available/finetuned)
    # ('BAAI/bge-base-financial', 'BGE-base-financial', 'Financial domain finetuned'),
]

# ============================================================================
# ⚡ SPEED OPTIMIZATION SETTINGS
# ============================================================================
SAMPLE_QUERIES_PER_DATASET = 30  # Number of queries to sample
MAX_CORPUS_SAMPLE_SIZE = 500     # Max corpus size after smart sampling
USE_SMART_SAMPLING = True        # Include all relevant docs
MAX_TEXT_LENGTH = 512            # Truncate long texts for embedding
BATCH_SIZE = 32                  # Embedding batch size

# ============================================================================
# LOCAL MODELS (auto-detected)
# ============================================================================
LOCAL_MODELS_DIR = Path('../../models')


def get_embedding_models():
    """Get all embedding models including local finetuned ones"""
    models = EMBEDDING_MODELS.copy()
    
    # Check for local finetuned models
    if LOCAL_MODELS_DIR.exists():
        for model_dir in LOCAL_MODELS_DIR.iterdir():
            if model_dir.is_dir() and 'financerag' in model_dir.name.lower():
                models.append(
                    (str(model_dir), model_dir.name, 'Local finetuned model')
                )
    
    return models


def print_config():
    """Print configuration summary"""
    models = get_embedding_models()
    
    print(f"\n📊 Configuration:")
    print(f"   Datasets: {len(DATASETS)}")
    print(f"   Embedding models: {len(models)}")
    print(f"\n⚡ Speed optimizations:")
    print(f"   Queries per dataset: {SAMPLE_QUERIES_PER_DATASET}")
    print(f"   Max corpus sample: {MAX_CORPUS_SAMPLE_SIZE}")
    print(f"   Smart sampling: {USE_SMART_SAMPLING}")
    print(f"   Max text length: {MAX_TEXT_LENGTH}")
    print(f"\n🤖 Models to evaluate:")
    for i, (model_name, display_name, desc) in enumerate(models, 1):
        print(f"   [{i}] {display_name}: {desc}")
