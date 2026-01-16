# MultiHeirtt Corpus Preprocessing

Preprocessing pipeline để chuyển đổi corpus MultiHeirtt sang format dễ hiểu hơn cho embedding models.

## 📊 Vấn đề

Từ phân tích ở các notebook trước:
- **50.3% queries** có NDCG=0 (hoàn toàn thất bại trong retrieval)
- Tables trong corpus có format markdown khó match với queries
- Vocabulary mismatch giữa query và documents

**Ví dụ vấn đề:**
```
Query: "What was the investment return in 2006?"

Document gốc (khó match):
| | Years Ended December 31, |
| | 2006 | 2005 |
| Investment return | $192 | $-26 |
```

## ✅ Giải pháp

Chuyển tables sang format tự nhiên:

```
Document sau preprocessing:
Investment return: Years Ended December 31, - 2006=$192, Years Ended December 31, - 2005=$-26
```

## 🚀 Cách sử dụng

### Option 1: Jupyter Notebook

Mở và chạy `preprocess_corpus.ipynb`

### Option 2: Command Line

```bash
# Linearized mode (recommended)
python preprocess_multiheirtt.py --mode linearized

# Augmented mode (original + linearized)
python preprocess_multiheirtt.py --mode augmented

# Row chunks mode (granular)
python preprocess_multiheirtt.py --mode row_chunks
```

## 📁 Output Files

| File | Mô tả | Use case |
|------|-------|----------|
| `multiheirtt_corpus_linearized.jsonl` | Tables → văn bản | **Recommended** cho embedding |
| `multiheirtt_corpus_augmented.jsonl` | Original + linearized | Hybrid retrieval, fallback |
| `multiheirtt_corpus_row_chunks.jsonl` | Mỗi row = 1 document | Fine-grained lookup |

## 📝 Preprocessing Modes

### 1. Linearized (Recommended)

Thay thế tables bằng văn bản linearized:

```
Before: | Revenue | $100 | $120 |
After:  Revenue: 2022=$100, 2023=$120
```

**Ưu điểm:** Giữ nguyên số lượng documents, embedding model hiểu tốt hơn

### 2. Augmented

Giữ document gốc + thêm linearized tables như documents riêng:

```
Document 1: [Original]
Document 2: [Linearized table 1]
Document 3: [Linearized table 2]
```

**Ưu điểm:** Coverage tốt hơn, có fallback

### 3. Row Chunks

Tách mỗi row trong table thành document riêng:

```
Document 1: [Original]
Document 2: Revenue: 2022=$100, 2023=$120
Document 3: Expenses: 2022=$80, 2023=$90
```

**Ưu điểm:** Matching chính xác cho queries cụ thể

## 🎯 Recommendation

1. **Bắt đầu với `linearized`** - tốt nhất cho embedding-based retrieval
2. Nếu recall thấp, thử `augmented`
3. Dùng `row_chunks` cho fine-grained matching

## 📈 Expected Improvement

Dựa trên phân tích:
- BM25 thuần: ~25% recovery rate
- Với preprocessing: Expected **40-50%** recovery rate
- Kết hợp fine-tuned embeddings: Expected **60-70%** recovery rate

---

## 🚀 Full Pipeline Notebook

**`final_pipeline_with_preprocessing.ipynb`** - Pipeline hoàn chỉnh:

1. **MultiHeirtt**: Áp dụng table preprocessing (linearize + row chunks)
2. **Các datasets khác**: Dùng pre-chunked corpus từ `data/chunked_corpus/`
3. **Hybrid retrieval**: Dense (E5 fine-tuned) + BM25
4. **Reranking**: BAAI/bge-reranker-v2-m3
5. **Output**: `submission_with_multiheirtt_preprocessing.csv`

### Chạy pipeline:
```python
# Mở và chạy notebook
final_pipeline_with_preprocessing.ipynb
```

### Đặc biệt cho MultiHeirtt:
- `use_preprocessing: True` → Tự động linearize tables
- `hybrid_alpha: 0.4` → 60% BM25 (tốt cho numerical matching)
- `top_k_retrieval: 200` → Retrieve nhiều hơn vì multi-doc queries
