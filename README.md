# 🚀 FinanceRAG - Financial Document Retrieval & Ranking

Hệ thống RAG (Retrieval-Augmented Generation) chuyên biệt cho tài liệu tài chính, sử dụng **semantic chunking**, **hybrid retrieval**, và **advanced reranking**.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start với Docker](#-quick-start-với-docker-recommended)
- [Cài đặt thủ công](#-cài-đặt-thủ-công)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Performance](#-performance)
- [Configuration](#-configuration)

---

## ✨ Features

### 🎯 Core Capabilities
- **Semantic Chunking**: Nhóm câu theo semantic similarity thay vì cắt cứng
- **Table-Aware Processing**: Phát hiện và bảo toàn cấu trúc bảng
- **Hybrid Retrieval**: Kết hợp Dense (vector) + BM25 (lexical)
- **Advanced Reranking**: Cross-encoder SOTA (BGE-reranker-v2-m3)
- **Dataset-Specific Optimization**: Tối ưu riêng cho từng dataset

### 📊 Supported Datasets
- ConvFinQA - Conversational financial QA
- FinanceBench - Financial benchmarking
- FinDER - Financial document retrieval
- FinQA - Financial question answering
- FinQABench - Financial QA benchmark
- MultiHeirTT - Hierarchical table reasoning
- TATQA - Table-text QA

### 🏆 Performance
- **Baseline NDCG@10**: 0.328
- **Optimized NDCG@10**: 0.50+ (improvement +52%+)
- **Semantic chunking adoption**: 5/7 datasets

---

## 🐳 Quick Start với Docker (Recommended)

### Prerequisites
- Docker Desktop (Windows/Mac) hoặc Docker Engine (Linux)
- Tối thiểu 8GB RAM
- 10GB disk space

### Option 1: CPU Only (Cho mọi máy)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd FinanceRAG

# 2. Build và start container
docker-compose up -d

# 3. Mở Jupyter Lab
# Browser tự động mở: http://localhost:8888
# Hoặc copy link từ terminal
```

### Option 2: With GPU (Nếu có NVIDIA GPU)

```bash
# Prerequisites: Cài đặt nvidia-docker2 trước
# Ubuntu/Linux: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Build và start với GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Verify GPU
docker exec -it financerag-notebook nvidia-smi
```

### 🎮 Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Execute commands in container
docker exec -it financerag-notebook bash

# Remove all data (reset)
docker-compose down -v
```

---

## 🛠️ Cài đặt thủ công

### 1. Requirements
- Python 3.10 hoặc 3.11
- pip 23.0+
- Git

### 2. Setup Virtual Environment

#### Windows
```bash
# Tạo virtual environment
python -m venv financerag_env

# Activate
financerag_env\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements_compatible.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('sentence-transformers OK')"
```

#### Linux/Mac
```bash
# Tạo virtual environment
python3 -m venv financerag_env

# Activate
source financerag_env/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements_compatible.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 3. Download Data

```bash
# Tạo thư mục data (nếu chưa có)
mkdir -p data

# Download datasets (hoặc copy từ Kaggle/competition source)
# Corpus files: data/<dataset>_corpus.jsonl/corpus.jsonl
# Queries files: data/<dataset>_queries.jsonl/queries.jsonl
# Qrels files: data/<dataset>_qrels.tsv
```

### 4. Start Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Hoặc Jupyter Notebook
jupyter notebook
```

---

## 📁 Project Structure

```
FinanceRAG/
│
├── 📂 data/                          # Datasets và preprocessed data
│   ├── *_corpus.jsonl/              # Document corpus
│   ├── *_queries.jsonl/             # Query sets
│   ├── *_qrels.tsv                  # Relevance judgments
│   └── chunked_corpus/              # Pre-chunked data (từ notebook 3)
│       ├── *_corpus_chunked_optimal.jsonl
│       └── best_chunking_config_per_dataset.json
│
├── 📂 financerag/                    # Core library
│   ├── common/                      # Shared utilities
│   ├── retrieval/                   # Retrieval models
│   │   ├── bm25.py                 # BM25 implementation
│   │   ├── dense.py                # Dense retrieval
│   │   └── sent_encoder.py        # Sentence encoder
│   ├── rerank/                      # Reranking models
│   │   └── cross_encoder.py        # Cross-encoder reranker
│   ├── generate/                    # Generation (optional)
│   └── tasks/                       # Dataset-specific tasks
│       ├── BaseTask.py
│       ├── ConvFinQATask.py
│       ├── FinanceBenchTask.py
│       └── ...
│
├── 📂 notebook/                      # Jupyter notebooks (Main workflows)
│   ├── 1_baseline/
│   │   └── 1. baseline.ipynb       # Baseline pipeline
│   ├── 2_quick_wins/
│   │   ├── 2. quick_wins_notebook.ipynb  # Quick improvements
│   │   ├── config.py               # Configuration
│   │   └── utils.py                # Shared utilities
│   ├── 3_chunking_evaluation/
│   │   ├── 3. chunking_evaluation.ipynb  # Evaluate chunking strategies
│   │   └── config.py
│   └── 4_improved_chunking/
│       ├── 4. improved_chunking_pipeline.ipynb  # Production pipeline
│       ├── config.py               # Dataset-specific config
│       └── utils.py
│
├── 📂 docker/                        # Docker configuration
│   ├── Dockerfile                  # Main Dockerfile
│   ├── docker-compose.yml          # Docker Compose (CPU)
│   ├── docker-compose.gpu.yml      # GPU support
│   └── .dockerignore
│
├── 📄 requirements.txt              # Python dependencies (original)
├── 📄 requirements_compatible.txt   # Compatible versions (use this!)
├── 📄 README.md                     # This file
└── 📄 LICENSE

```

---

## 📖 Usage Guide

### Workflow Overview

```
1️⃣ Baseline (notebook 1)
   ↓
2️⃣ Quick Wins (notebook 2) - Basic improvements
   ↓
3️⃣ Chunking Evaluation (notebook 3) - Find optimal chunking per dataset
   ↓
4️⃣ Production Pipeline (notebook 4) - Final submission with all optimizations
```

### 1️⃣ Baseline Pipeline

```python
# Open: notebook/1_baseline/1. baseline.ipynb

# Loads data, retrieves with BGE-M3, reranks, generates submission
# NDCG@10: ~0.328 (baseline)
```

### 2️⃣ Quick Wins (Recommended starting point)

```python
# Open: notebook/2_quick_wins/2. quick_wins_notebook.ipynb

# Features:
# - Better models (BGE-large, BGE-reranker-v2-m3)
# - Table-aware chunking
# - Hybrid retrieval (Dense + BM25)
# - Local evaluation with NDCG@10

# Expected NDCG@10: ~0.40-0.50
```

### 3️⃣ Chunking Evaluation (Advanced)

```python
# Open: notebook/3_chunking_evaluation/3. chunking_evaluation.ipynb

# Systematically evaluates:
# - Semantic chunking (NEW!)
# - Recursive chunking
# - Fixed-size chunking
# - Table-specific methods

# Output:
# - best_chunking_config_per_dataset.json
# - Pre-chunked corpora for production
```

### 4️⃣ Production Pipeline (Best Performance)

```python
# Open: notebook/4_improved_chunking/4. improved_chunking_pipeline.ipynb

# Uses optimal chunking from notebook 3:
# - Semantic chunking for most datasets
# - Dataset-specific retrieval parameters
# - Hybrid search with tuned alpha

# Expected NDCG@10: ~0.50-0.58+
```

---

## 🎯 Performance

## 🎯 Performance

### Evaluation Metrics (30% qrels)

| Dataset | Baseline | Optimized | Improvement | Chunking Method |
|---------|----------|-----------|-------------|-----------------|
| ConvFinQA | 0.328 | 0.659 | +101% 🚀 | Semantic (2000/0.6) |
| FinanceBench | 0.328 | 0.794 | +142% 🚀 | Semantic (1000/0.7) |
| FinDER | 0.661 | 0.705 | +6.6% 🟢 | Recursive (512/50) |
| FinQA | 0.328 | 0.628 | +91% 🚀 | Semantic (2500/0.65) |
| FinQABench | 0.868 | 0.868 | 0% 🟡 | Preserve Tables |
| MultiHeirTT | 0.163 | 0.213 | +31% 🟢 | Semantic (3000/0.65) |
| TATQA | 0.372 | 0.534 | +44% 🟢 | Semantic (2000/0.7) |
| **Average** | **0.328** | **0.50+** | **+52%** 🏆 | **Dataset-specific** |

### Key Insights

✅ **Semantic chunking** thắng 5/7 datasets  
✅ **Table-aware processing** critical cho FinQABench  
✅ **Hybrid retrieval** cải thiện numerical reasoning (MultiHeirTT, TATQA)  
✅ **Dataset-specific tuning** quan trọng - không có "one size fits all"

---

## ⚙️ Configuration

### Chunking Strategies

```python
# config.py example
CONFIG = {
    'use_prechunked': True,  # Load pre-optimized chunks
    'chunked_corpus_dir': '../../data/chunked_corpus',
    
    # Fallback chunking (if pre-chunked not available)
    'chunking_method': 'semantic',  # 'semantic', 'recursive', 'fixed'
    'chunk_size': 1500,             # characters
    'chunk_overlap': 300,           # characters
    'chunk_aggregation': 'max',     # 'max', 'mean', 'weighted'
}
```

### Dataset-Specific Overrides

```python
# Numerical reasoning datasets need more candidates
DATASET_SPECIFIC_CONFIG = {
    'multiheirtt': {
        'top_k_retrieval': 200,  # vs default 100
        'top_k_rerank': 80,      # vs default 50
        'hybrid_alpha': 0.4,     # 60% BM25, 40% Dense
    },
    'tatqa': {
        'top_k_retrieval': 150,
        'hybrid_alpha': 0.5,     # 50-50 balanced
    },
}
```

### Model Selection

```python
# Best models (from evaluation)
MODELS = {
    'embedding': 'BAAI/bge-large-en-v1.5',  # or 'intfloat/e5-large-v2'
    'reranker': 'BAAI/bge-reranker-v2-m3',  # SOTA cross-encoder
}
```

---

## 🔧 Advanced Usage

### Using Task Classes (Original API)

```python
# Import task
from financerag.tasks import FinDER
finder_task = FinDER()

# Setup retriever
from sentence_transformers import SentenceTransformer
from financerag.retrieval import SentenceTransformerEncoder, DenseRetrieval

model = SentenceTransformer('intfloat/e5-large-v2')
encoder = SentenceTransformerEncoder(
    q_model=model,
    doc_model=model,
    query_prompt='query: ',
    doc_prompt='passage: '
)
retriever = DenseRetrieval(model=encoder)

# Retrieve
results = finder_task.retrieve(retriever=retriever)

# Rerank
from financerag.rerank import CrossEncoderReranker
reranker = CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L-12-v2')
reranked = finder_task.rerank(reranker, results, top_k=100)

# Save
finder_task.save_results(output_dir='./results')
```

### Custom Chunking

```python
from notebook.utils import chunk_corpus, load_jsonl

# Load corpus
corpus = load_jsonl('data/financebench_corpus.jsonl/corpus.jsonl')

# Apply semantic chunking
chunked = chunk_corpus(
    corpus, 
    method='semantic',
    chunk_size=1000,
    overlap=0.7,  # similarity threshold for semantic
    dataset_name='financebench'
)

# Save
import json
with open('output/chunked_corpus.jsonl', 'w') as f:
    for doc in chunked:
        f.write(json.dumps(doc) + '\n')
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch sizes in config.py
CONFIG = {
    'embed_batch_size': 8,   # default: 16
    'rerank_batch_size': 8,  # default: 16
}
```

**2. CUDA not available**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Slow retrieval**
```python
# Use smaller corpus sample for testing
CONFIG = {
    'use_smart_sampling': True,
    'max_corpus_sample_size': 3000  # default: full corpus
}
```

**4. Missing pre-chunked data**
```bash
# Run notebook 3 first to generate optimal chunks
# Or set use_prechunked=False in config to chunk on-the-fly
```

---

## 📚 Resources

### Documentation
- [Chunking Evaluation Report](data/chunked_corpus/dataset_specific_evaluation_report.txt)
- [Best Chunking Configs](data/chunked_corpus/best_chunking_config_per_dataset.json)
- [Notebook Tutorials](notebook/)

### Models
- [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) - Embedding model
- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) - Reranking model
- [E5-large-v2](https://huggingface.co/intfloat/e5-large-v2) - Alternative embedding

### Papers
- [BGE: Retrieval Augmented Generation](https://arxiv.org/abs/2309.07597)
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Competition organizers and dataset providers
- Hugging Face for model hosting
- BAAI and E5 teams for excellent models
- Open source community

---

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact maintainers

---

**Happy Retrieving! 🚀**