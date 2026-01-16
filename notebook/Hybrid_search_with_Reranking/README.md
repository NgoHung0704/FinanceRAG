# Hybrid Search & Reranking Evaluation

Thư mục này chứa các notebook để đánh giá và tối ưu hóa **hybrid search alpha** và **reranker models** cho FinanceRAG.

## 📋 Notebooks

### 1. [hybrid_alpha_evaluation.ipynb](1.%20hybrid_alpha_evaluation.ipynb)
Test hybrid_alpha tốt nhất cho từng dataset.

**Chức năng:**
- Test alpha từ 0.1 đến 0.9 (step 0.1)
- Tính NDCG@10 cho mỗi giá trị alpha
- Tìm alpha tối ưu per dataset
- Phân loại: Dense-heavy, Balanced, hoặc BM25-heavy
- Visualizations: line plots, heatmap

**Output:**
- `output_hybrid_rerank_eval/alpha_evaluation_results.csv`
- `output_hybrid_rerank_eval/best_alpha_per_dataset.csv`
- `output_hybrid_rerank_eval/alpha_vs_ndcg_per_dataset.png`
- `output_hybrid_rerank_eval/alpha_heatmap.png`

### 2. [reranker_model_evaluation.ipynb](2.%20reranker_model_evaluation.ipynb)
Test reranker models tốt nhất cho toàn bộ datasets.

**Chức năng:**
- Test 5 reranker models (BGE + MS-MARCO cross-encoders)
- Đo NDCG@10 và reranking time
- So sánh speed vs accuracy
- Tìm best reranker overall và per dataset
- Visualizations: bar charts, scatter, heatmap

**Output:**
- `output_hybrid_rerank_eval/reranker_evaluation_results.csv`
- `output_hybrid_rerank_eval/reranker_ranking.csv`
- `output_hybrid_rerank_eval/best_reranker_per_dataset.csv`
- `output_hybrid_rerank_eval/reranker_performance_comparison.png`

## 🔧 Configuration

File [config.py](config.py) chứa tất cả cấu hình:

```python
# Alpha values to test
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Reranker models to test
RERANKER_MODELS = [
    ('BAAI/bge-reranker-base', 'BGE-reranker-base', ...),
    ('BAAI/bge-reranker-large', 'BGE-reranker-large', ...),
    # ...
]

# Sampling settings
SAMPLE_QUERIES_PER_DATASET = 30  # Queries per dataset
MAX_CORPUS_SAMPLE_SIZE = 500     # Max corpus size

# Retrieval parameters
TOP_K_RETRIEVAL = 100
TOP_K_RERANK = 50
TOP_K_FINAL = 10
```

## 🚀 Usage

### Prerequisites

Đảm bảo đã:
1. Chạy notebook 4 để tạo pre-chunked corpus
2. Install dependencies: `pip install sentence-transformers faiss-cpu FlagEmbedding rank-bm25`

### Running Notebooks

**Notebook 1 - Alpha Evaluation:**
```bash
# Open in Jupyter/VS Code
1. hybrid_alpha_evaluation.ipynb

# Run all cells (Shift+Enter hoặc Run All)
# Kết quả sẽ lưu vào output_hybrid_rerank_eval/
```

**Notebook 2 - Reranker Evaluation:**
```bash
# Open in Jupyter/VS Code
2. reranker_model_evaluation.ipynb

# Run all cells
# Kết quả sẽ lưu vào output_hybrid_rerank_eval/
```

## ⚠️ Troubleshooting

### Kernel Crash / Out of Memory

**Triệu chứng:**
- Kernel crashes với exit code 3221225477
- "CUDA out of memory" error
- System freezes

**Giải pháp:**

1. **Giảm sample size** trong `config.py`:
   ```python
   SAMPLE_QUERIES_PER_DATASET = 20  # từ 30 xuống 20
   TOP_K_RETRIEVAL = 50             # từ 100 xuống 50
   ```

2. **Chạy trên CPU** (chậm hơn nhưng ổn định):
   ```python
   # Trong cell "Setup and Imports"
   device = 'cpu'  # Force CPU
   ```

3. **Giảm số reranker models** test:
   ```python
   # Trong config.py, chỉ test 2-3 models quan trọng
   RERANKER_MODELS = [
       ('BAAI/bge-reranker-v2-m3', 'BGE-reranker-v2-m3', ...),
       ('cross-encoder/ms-marco-MiniLM-L-6-v2', 'MiniLM-L6-cross', ...),
   ]
   ```

4. **Clear memory giữa các runs**:
   ```python
   import gc
   import torch
   
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   gc.collect()
   ```

5. **Restart kernel** giữa notebook 1 và notebook 2

### Import Errors

**Lỗi:** `ModuleNotFoundError: No module named 'FlagEmbedding'`

**Giải pháp:**
```bash
pip install FlagEmbedding
pip install rank-bm25
pip install faiss-cpu  # hoặc faiss-gpu nếu có CUDA
```

### File Not Found

**Lỗi:** `Pre-chunked corpus not found`

**Giải pháp:**
- Chạy notebook 4 (improved_chunking_pipeline) trước
- Đảm bảo file tồn tại: `../../data/chunked_corpus/*_corpus_chunked_optimal.jsonl`

## 📊 Expected Results

**Notebook 1 (Alpha Evaluation):**
- Mỗi dataset sẽ có alpha tối ưu riêng
- Datasets với numerical reasoning (MultiHeirTT, TATQA) thường prefer alpha thấp (< 0.5, BM25-heavy)
- Datasets với semantic search thường prefer alpha cao (> 0.6, Dense-heavy)

**Notebook 2 (Reranker Evaluation):**
- BGE-reranker-v2-m3 thường là best overall
- BGE-reranker-large tốt nhưng chậm hơn
- Cross-encoders nhỏ (MiniLM) nhanh nhưng accuracy thấp hơn

## 🎯 Next Steps

Sau khi có kết quả:
1. Update `DATASET_SPECIFIC_CONFIG` trong notebook 4 với optimal alpha
2. Update reranker model trong final pipeline
3. Chạy notebook 3 (sẽ được tạo sau) để tích hợp kết quả

## 💡 Tips

- **Chạy notebook 1 trước** để tìm alpha tối ưu
- **Sử dụng alpha từ notebook 1** làm default cho notebook 2
- **Monitor GPU memory** bằng `nvidia-smi` (Linux/WSL) hoặc Task Manager (Windows)
- **Save checkpoints** bằng cách export results thường xuyên
- **Batch processing**: Nếu gặp lỗi, có thể chạy từng dataset riêng lẻ

## 📞 Support

Nếu gặp vấn đề:
1. Check notebook 0 (embedding_model_evaluation) - cùng structure
2. Review utils.py functions
3. Verify pre-chunked data exists
4. Check GPU memory availability
