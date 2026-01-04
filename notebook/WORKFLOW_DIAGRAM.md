# 📊 Dataset-Specific Chunking Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATASET-SPECIFIC CHUNKING PIPELINE                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  📚 Input Datasets  │
│  (7 datasets)       │
└──────────┬──────────┘
           │
           ├─── 1. TATQA (100% tables, avg 2433 chars)
           ├─── 2. MultiHeirTT (67% tables, hierarchical, avg 2956 chars)
           ├─── 3. ConvFinQA (100% tables+text, very long avg 4526 chars)
           ├─── 4. FinQA (100% tables+text, avg 4394 chars)
           ├─── 5. FinanceBench (0% tables, pure text, avg 1359 chars)
           ├─── 6. FINDER (0% tables, very short avg 576 chars, large corpus)
           └─── 7. FinQABench (30% tables, mixed, small corpus 92 docs)
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   🔍 DATASET CHARACTERIZATION                           │
│  - Detect table presence (%, structure type)                            │
│  - Calculate document lengths (avg, distribution)                       │
│  - Identify content type (pure table / hybrid / pure text)              │
│  - Corpus size analysis                                                 │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              🧪 DATASET-SPECIFIC STRATEGY SELECTION                     │
│                                                                          │
│  TABLE-HEAVY DATASETS:                                                  │
│  ├─ TATQA, MultiHeirTT, ConvFinQA, FinQA                               │
│  └─ Test: no_chunking, preserve_tables(3000), preserve_tables(4096)    │
│                                                                          │
│  TEXT-HEAVY DATASETS:                                                   │
│  ├─ FinanceBench                                                        │
│  └─ Test: recursive(512), recursive(768), recursive(1024)              │
│                                                                          │
│  SHORT-DOC DATASETS:                                                    │
│  ├─ FINDER, FinQABench                                                 │
│  └─ Test: no_chunking, recursive(512), preserve_tables(2048)           │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           │  For each dataset:
           │  ┌──────────────────────────────────────────────┐
           │  │  FOR EACH CHUNKING STRATEGY:                 │
           │  │                                              │
           │  │  1. Load corpus + queries + qrels            │
           │  │  2. Apply chunking method                    │
           │  │  3. Encode chunks with embedder              │
           │  │  4. Retrieve top-k chunks per query          │
           │  │  5. Aggregate chunks → documents             │
           │  │  6. Compute NDCG@10                          │
           │  │  7. Store results                            │
           │  └──────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    📊 EVALUATION & COMPARISON                           │
│                                                                          │
│  For each dataset:                                                      │
│  ├─ Compare all tested strategies                                       │
│  ├─ Select best strategy (max NDCG@10)                                  │
│  └─ Calculate improvement vs baseline                                   │
│                                                                          │
│  Across all datasets:                                                   │
│  ├─ Average NDCG@10 with optimal per-dataset config                     │
│  ├─ Compare with uniform chunking baseline                              │
│  └─ Analyze method effectiveness                                        │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     💾 OUTPUT GENERATION                                │
│                                                                          │
│  1. best_chunking_config_per_dataset.json                               │
│     {                                                                    │
│       "tatqa": {"method": "no_chunking", "ndcg_10": 0.4935},           │
│       "financebench": {"method": "recursive", "chunk_size": 512, ...}   │
│     }                                                                    │
│                                                                          │
│  2. dataset_chunking_method_mapping.json                                │
│     → Simplified config for production                                  │
│                                                                          │
│  3. Chunked corpora: tatqa_corpus_chunked_optimal.jsonl, ...           │
│     → Pre-chunked with optimal strategy                                 │
│                                                                          │
│  4. Visualizations:                                                     │
│     ├─ Performance by dataset (bar chart)                               │
│     ├─ Improvement vs baseline (bar chart)                              │
│     ├─ Dataset × Method heatmap                                         │
│     ├─ Method performance comparison                                    │
│     └─ Strategy distribution                                            │
│                                                                          │
│  5. Reports:                                                            │
│     ├─ dataset_specific_evaluation_report.txt                           │
│     └─ dataset_specific_chunking_results.csv                            │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   🚀 PRODUCTION DEPLOYMENT                              │
│                                                                          │
│  Step 1: Load chunking configuration                                    │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ with open('dataset_chunking_method_mapping.json') as f:       │     │
│  │     config = json.load(f)                                     │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Step 2: For each dataset, load pre-chunked corpus                      │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ corpus_file = f'{dataset}_corpus_chunked_optimal.jsonl'       │     │
│  │ chunked_corpus = load_jsonl(corpus_file)                      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Step 3: Retrieve with chunks + aggregate to documents                  │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ # Encode query + chunks                                       │     │
│  │ similarities = cosine_similarity(query_emb, chunk_embs)       │     │
│  │                                                               │     │
│  │ # Aggregate by original_id (MAX score)                        │     │
│  │ doc_scores[chunk['original_id']] = max(...)                   │     │
│  │                                                               │     │
│  │ # Return top-k documents                                      │     │
│  │ return sorted(doc_scores.items(), ...)[:k]                    │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ✅ Expected Improvements:                                              │
│  ├─ FinanceBench: +114% (0.34 → 0.74)                                  │
│  ├─ TATQA: +3.5% (0.48 → 0.49)                                         │
│  ├─ MultiHeirTT: +100-170% (0.15 → 0.30-0.40)                          │
│  └─ Overall: +15-27% average NDCG@10                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Decision Points

### 1. Table Detection
```
Is document a table?
├─ YES → Use preserve_tables or no_chunking
└─ NO → Use recursive chunking
```

### 2. Chunk Size Selection
```
Document length?
├─ <1000 chars → no_chunking or 512 chunks
├─ 1000-3000 chars → 512-1024 chunks  
└─ >3000 chars (tables) → 2048-4096 chunks or no_chunking
```

### 3. Aggregation Strategy
```
Multiple chunks per document?
├─ YES → Aggregate using MAX score (best empirical results)
└─ NO → Use chunk score directly
```

## 📈 Performance Expectations

| Dataset | Current | Expected | Gain | Strategy |
|---------|---------|----------|------|----------|
| FinanceBench | 0.3439 | 0.7362 | +114% | recursive(512) ✅ PROVEN |
| TATQA | 0.4768 | 0.4935 | +3.5% | no_chunking ✅ PROVEN |
| MultiHeirTT | 0.1467 | 0.30-0.40 | +100-170% | preserve_tables(4096) 🎯 TARGET |
| ConvFinQA | 0.4858 | 0.50-0.53 | +3-9% | preserve_tables(2048) |
| FinQA | 0.4570 | 0.47-0.50 | +3-9% | preserve_tables(2048) |
| FINDER | 0.3953 | 0.39-0.42 | 0-6% | no_chunking or recursive(512) |
| FinQABench | 0.8662 | 0.86-0.88 | 0-2% | no_chunking (already optimal) |
| **OVERALL** | **0.4949** | **0.57-0.63** | **+15-27%** | **Per-dataset config** |

---

**Note**: This is an ITERATIVE process. Run evaluation → analyze results → refine strategies → re-run if needed.
