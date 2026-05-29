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

## 💹 Interactive RAG App (NEW)

Ngoài pipeline notebook (thi đấu, xuất CSV theo NDCG@10), dự án giờ có một **ứng dụng RAG deploy được**: gõ câu hỏi tài chính → nhận **câu trả lời do LLM sinh ra, có trích dẫn**, dựa trên các đoạn văn bản được truy hồi.

**Kiến trúc: `Retrieve → Rerank → Generate`** — giữ nguyên bộ truy hồi mạnh của dự án làm xương sống:

```
question ─▶ Hybrid retrieval (dense E5/BGE + BM25)
         ─▶ Cross-encoder rerank (BGE-reranker-v2-m3)
         ─▶ OpenAI answer with citations [1][2]
```

> **Why not LightRAG / GraphRAG?** Corpus ở đây nặng về **bảng số liệu** (TAT-QA, MultiHierTT) — đúng điểm yếu của graph-RAG (entity extraction kém trên bảng), lại tốn LLM để index cả corpus (10K+ docs) và không phục vụ metric NDCG@10. Bộ hybrid retriever hiện tại mạnh hơn cho dữ liệu này; LLM chỉ thêm ở tầng *generate*. (Có thể bổ sung graph mode cho dataset narrative sau.)

### Run locally

```bash
# 1. Install app deps (retrieval + LLM + Streamlit)
make install-app                      # hoặc: pip install -r requirements-app.txt

# 2. (Optional) enable AI answers — không set key vẫn chạy ở chế độ retrieval-only
export OPENAI_API_KEY=sk-...          # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."

# 3. (Optional) pre-build indexes so the first query is fast
make app-build-index                  # hoặc: python scripts/build_index.py finder

# 4. Launch the UI  ->  http://localhost:8501
make app                              # hoặc: streamlit run app/streamlit_app.py
```

### Run with Docker

```bash
export OPENAI_API_KEY=sk-...          # optional
make docker-app-up                    # -> http://localhost:8501
```

### Evaluate retrieval quality (NDCG@10, dùng chung code với app)

```bash
python scripts/run_eval.py            # tất cả datasets
python scripts/run_eval.py --no-rerank  # ablation: tắt reranker
```

### Thử nghiệm: chẩn đoán + so sánh LightRAG

```bash
# 1) Chẩn đoán retrieval-vs-ranking (RẺ, không tốn API)
python scripts/diagnose.py
#    Rec@100 cao + NDCG@10 thấp  -> lỗi XẾP HẠNG (sửa reranker/alpha); LightRAG không giúp
#    Rec@100 thấp                -> lỗi RETRIEVAL; mới đáng đổi paradigm

# 2) So sánh hybrid vs LightRAG, NDCG@10 retrieve-only (cần OPENAI_API_KEY)
make install-lightrag
export OPENAI_API_KEY=sk-...
python scripts/compare_lightrag.py    # finqabench + financebench + mẫu multiheirtt
```

> ⚠️ LightRAG dựng knowledge graph bằng LLM lúc index (tốn phí, chậm) và mạnh nhất cho
> văn bản tự sự multi-hop — **yếu ở bảng số** (multiheirtt/tatqa). So sánh là retrieve-only;
> LightRAG dùng embedding OpenAI nên chênh lệch trộn cả "graph vs không" lẫn "embedding khác".
> Coi đây là phép thử thực nghiệm để xem số thật, không phải ablation thuần.

### New code layout

```
financerag_app/        # clean, unified library (lazy heavy imports)
├── config.py          # single source of truth: per-dataset chunking/alpha/top_k
├── data.py            # stdlib-only loaders (corpus/queries/qrels/prechunked)
├── chunking.py        # table-aware chunking (recursive/preserve_tables/semantic)
├── retriever.py       # HybridRetriever: FAISS dense + BM25, disk-cached
├── reranker.py        # CrossEncoderReranker (BGE-reranker-v2-m3)
├── generator.py       # OpenAIGenerator: grounded answers + citations
├── pipeline.py        # RAGPipeline: retrieve → rerank → generate
├── evaluate.py        # NDCG@10 / Recall / MRR (pure Python)
└── lightrag_retriever.py  # optional graph-RAG (LightRAG) for comparison
app/streamlit_app.py   # the interface
scripts/               # build_index.py, run_eval.py, diagnose.py, compare_lightrag.py
tests/test_core.py     # pure-Python tests (no ML stack required)
```

Config overrides via env: `FINRAG_EMBEDDING_MODEL`, `FINRAG_RERANKER_MODEL`, `FINRAG_LLM_MODEL`, `FINRAG_DATA_DIR`, `FINRAG_CACHE_DIR`, `OPENAI_API_KEY`.

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
pip install -r requirements.txt

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
pip install -r requirements.txt

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

### 4. Run the app

```bash
# (optional) enable AI answers
export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."

# launch the Streamlit interface -> http://localhost:8501
streamlit run app/streamlit_app.py
```

---

## 📁 Project Structure

```
FinanceRAG/
│
├── 📂 financerag_app/                # Unified RAG library (the app core)
│   ├── config.py                    # Single source of truth (per-dataset tuning)
│   ├── data.py                      # Corpus/queries/qrels/prechunked loaders
│   ├── chunking.py                  # Table-aware chunking
│   ├── retriever.py                 # HybridRetriever (FAISS dense + BM25, cached)
│   ├── reranker.py                  # CrossEncoderReranker (BGE-reranker-v2-m3)
│   ├── generator.py                 # OpenAIGenerator (grounded answers + citations)
│   ├── pipeline.py                  # RAGPipeline: retrieve → rerank → generate
│   └── evaluate.py                  # NDCG@10 / Recall / MRR
│
├── 📂 app/
│   └── streamlit_app.py             # The web interface
│
├── 📂 scripts/
│   ├── build_index.py               # Pre-build & cache retrieval indexes
│   └── run_eval.py                  # NDCG@10 evaluation (shares app code)
│
├── 📂 tests/
│   └── test_core.py                 # Pure-Python tests (no ML stack needed)
│
├── 📂 data/                          # Datasets
│   ├── *_corpus.jsonl/corpus.jsonl  # Document corpus
│   ├── *_queries.jsonl/queries.jsonl  # Query sets
│   ├── *_qrels.tsv                  # Relevance judgments
│   └── chunked_corpus/              # Pre-chunked corpora used by the retriever
│       └── *_corpus_chunked_optimal.jsonl
│
├── 📂 financerag/                    # Original competition library (legacy)
│   ├── retrieval/  rerank/  generate/  tasks/  common/
│
├── 📂 models/                        # Fine-tuned embedding model (git-ignored)
│
├── 🐳 Dockerfile.app                 # Image for the Streamlit app
├── 🐳 docker-compose.yml             # `app` service + (legacy) jupyter service
├── 📄 requirements-app.txt           # App runtime deps (use this for the app)
├── 📄 requirements.txt               # Full/legacy deps
├── 📄 README.md
└── 📄 LICENSE

```

---

## 📖 Usage Guide

### Workflow

```
1️⃣ build_index.py  →  cache FAISS + BM25 per dataset
   ↓
2️⃣ streamlit_app   →  ask questions, get answers + cited passages
   ↓
3️⃣ run_eval.py     →  measure NDCG@10 (same retrieve+rerank code as the app)
```

### Use the library directly

```python
from financerag_app.config import AppConfig
from financerag_app.pipeline import RAGPipeline

pipe = RAGPipeline(AppConfig())
result = pipe.query("finder", "What are Microsoft's main product segments?", top_k=10)

print(result.answer)                       # LLM answer with [1][2] citations (if key set)
for p in result.passages:                  # ranked supporting passages
    print(p.rank, p.doc_id, round(p.score, 3), p.title)
```

### Tune per-dataset behaviour

`financerag_app/config.py` holds the distilled per-dataset settings (chunking
method/size, hybrid α, retrieval depths). Override models/keys via env vars
(`FINRAG_EMBEDDING_MODEL`, `FINRAG_LLM_MODEL`, `OPENAI_API_KEY`, …) — no code edits.

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
from financerag_app.data import load_corpus
from financerag_app.chunking import chunk_corpus

corpus = load_corpus('financebench', 'data')

# method: recursive | preserve_tables | semantic | fixed | none
# (for 'semantic', overlap is the similarity threshold)
chunks, chunk_to_doc = chunk_corpus(corpus, method='semantic', size=1000, overlap=0.7)
print(len(chunks), 'chunks')
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
# The retriever falls back to on-the-fly chunking automatically.
# To force it, set use_prechunked=False on AppConfig (or it triggers when
# data/chunked_corpus/<dataset>_corpus_chunked_optimal.jsonl is absent).
```

---

## 📚 Resources

### Documentation
- [Per-dataset chunking methods](data/chunked_corpus/dataset_chunking_method_mapping.json)
- App library: [financerag_app/](financerag_app/) · Interface: [app/streamlit_app.py](app/streamlit_app.py)

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