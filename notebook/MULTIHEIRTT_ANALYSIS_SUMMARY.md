# 🔍 MULTIHEIRTT Query-Level Analysis Summary

**Analysis Date:** January 10, 2026  
**Current NDCG@10:** 0.1497 (worst performing dataset)

---

## 📊 Executive Summary

MultiHeirTT dataset có performance **rất thấp** so với các datasets khác:
- **50.3%** queries có NDCG = 0.0 (retrieval failure hoàn toàn)
- **53.1%** queries có NDCG < 0.1
- Chỉ **3.1%** queries có NDCG > 0.6
- **Không có** query nào đạt perfect score (1.0)

**Root Cause:** Dataset yêu cầu **numerical/tabular reasoning** mà current retrieval approach không handle tốt.

---

## 🔴 Problem Analysis

### 1. Query Characteristics (974 queries total, 292 in evaluation)

| Characteristic | Percentage | Impact |
|---------------|------------|--------|
| Has Numbers | 74.3% | ❌ Current approach struggles with exact number matching |
| Needs Calculation | 70.2% | ❌ Requires multi-document aggregation |
| Has Comparison | 33.6% | ❌ Needs understanding of "highest", "most", "largest" |
| Mentions Table | 30.5% | ⚠️ Tables are preserved but may need better linearization |
| Average Length | 18 words | ⚠️ Complex questions with multiple conditions |

### 2. Relevant Documents Distribution

- **Average relevant docs per query:** 4.6 documents
- **Problem:** Need to retrieve MULTIPLE relevant documents per query
- **Current top_k:** 200 → may not be enough for coverage

### 3. WORST Query Patterns (NDCG = 0.0)

**Pattern 1: Arithmetic Operations**
```
❌ "What's the sum of X and Y?"
❌ "What is the average of X in years where Y > 0?"
❌ "What was the total amount of A, B, and C?"
```
→ Needs exact number extraction + calculation

**Pattern 2: Temporal Comparisons**
```
❌ "What was the percentage change from 2008 to 2009?"
❌ "In the year with the most X, what is Y?"
❌ "Does X keep increasing each year between 2003 and 2004?"
```
→ Needs multi-document temporal reasoning

**Pattern 3: Conditional Queries**
```
❌ "What is the growing rate of X in the year with the most Y?"
❌ "What was the average X in sections where Y > Z?"
```
→ Multi-hop reasoning: find year → extract value → compute

### 4. BEST Query Patterns (NDCG > 0.6)

**Success Pattern: Specific Entity Names**
```
✅ "What is the sum of CET1 capital, Tier 1 capital in 2017?"
✅ "What is the annualized return for allegion plc during 2013-2017?"
```
→ Specific named entities help with retrieval

**Key Difference:**
- Worst queries: generic terms ("Other investment income", "Total reserves")
- Best queries: specific names ("CET1 capital", "allegion plc", "TRICARE")

---

## 💡 Actionable Recommendations

### 🔥 Priority 1: Improve Retrieval Coverage

**Problem:** 50.3% queries have zero relevant docs in top-10
**Solution:**

```python
MULTIHEIRTT_IMPROVED_CONFIG = {
    'top_k_retrieval': 500,    # ↑ from 200 (need to retrieve 4.6 docs on avg)
    'top_k_rerank': 100,       # ↑ from 80
    'hybrid_alpha': 0.3,       # ↓ from 0.4 (70% BM25 for exact matching)
}
```

**Expected Impact:** +20-30% NDCG (from 0.15 → 0.18-0.20)

### 🎯 Priority 2: Number-Aware Retrieval

**Problem:** 74.3% queries contain numbers, but embedding models don't handle them well
**Solutions:**

1. **Number Normalization:**
   ```python
   # Normalize numbers in text
   "2008" → "[YEAR_2008]"
   "15.5%" → "[PERCENTAGE]"
   "$1.5M" → "[MONEY]"
   ```

2. **Boost BM25 Weight:**
   - Current: 60% BM25
   - Recommended: **70-80% BM25** for MultiHeirTT
   - BM25 is better for exact token matching (numbers, years)

3. **Use ColBERT:**
   - Token-level matching instead of sentence-level
   - Better for numerical expressions

**Expected Impact:** +10-15% NDCG

### 📊 Priority 3: Table-Specific Processing

**Problem:** 30.5% queries mention tables, and tables contain the critical numerical data
**Solutions:**

1. **Improved Table Linearization:**
   ```
   Current: "| Header1 | Header2 | ..."
   Better: "Table: Header1 is X, Header2 is Y for Row1. ..."
   ```

2. **Table Metadata:**
   - Add table captions as separate searchable text
   - Index row/column headers separately

**Expected Impact:** +5-10% NDCG

### 🔍 Priority 4: Query Understanding

**Problem:** Queries are complex (avg 18 words, multi-hop reasoning)
**Solutions:**

1. **Query Reformulation with LLM:**
   ```
   Original: "What is X in the year with the most Y?"
   
   Reformulated:
   1. "What year has the highest Y?"
   2. "What is the value of X in [year]?"
   ```

2. **Extract Key Entities:**
   - Numbers: "2008", "2009"
   - Metrics: "percentage change", "average", "sum"
   - Entities: specific company/product names

3. **Query Expansion:**
   - Synonyms for financial terms
   - Different date formats

**Expected Impact:** +15-20% NDCG

---

## 🚀 Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)
- [x] **Analyze query patterns** → ✅ Done
- [ ] **Increase top_k to 500** → Re-run pipeline
- [ ] **Increase BM25 weight to 70%** → Modify config
- [ ] **Evaluate impact** → Compare with baseline

**Expected:** 0.1497 → **0.20-0.22** (+35-47%)

### Phase 2: Medium Effort (1-2 days)
- [ ] **Number normalization** → Preprocessing script
- [ ] **Table linearization** → Improve chunking
- [ ] **Query expansion** → Add synonym dictionary
- [ ] **Re-evaluate** → Check improvement

**Expected:** 0.20 → **0.25-0.28** (+67-87%)

### Phase 3: Advanced (3-5 days)
- [ ] **LLM query reformulation** → Use GPT-4/Claude
- [ ] **ColBERT for token-level matching** → New model
- [ ] **Multi-stage retrieval** → Cascade approach
- [ ] **Fine-tune embeddings** → MultiHeirTT-specific

**Expected:** 0.25 → **0.35-0.40** (+134-167%)

---

## 📈 Comparison with Other Datasets

| Dataset | NDCG@10 | Characteristics |
|---------|---------|-----------------|
| **FinQABench** | 0.8662 | ✅ Well-structured, clear queries |
| **FinanceBench** | 0.7332 | ✅ Document-level, less calculation |
| **TATQA** | 0.5048 | ⚠️ Tables, but better than MultiHeirTT |
| **ConvFinQA** | 0.4866 | ⚠️ Conversational, needs context |
| **FinQA** | 0.4601 | ⚠️ Similar to MultiHeirTT but better |
| **FinDER** | 0.3903 | ⚠️ Relation extraction, difficult |
| **MultiHeirTT** | **0.1497** | ❌ **Hierarchical tables + multi-hop** |

**Key Insight:** MultiHeirTT is uniquely challenging because it combines:
1. Hierarchical table structures
2. Numerical reasoning
3. Multi-hop questions
4. Temporal comparisons

Other datasets have 1-2 of these, but MultiHeirTT has **all 4**.

---

## 🎯 Success Metrics

### Current State
- NDCG@10: 0.1497
- Zero-score queries: 50.3%
- Average relevant retrieved (top-10): ~0.5 out of 4.6

### Target (Phase 1)
- NDCG@10: **0.20+** (+35%)
- Zero-score queries: **<40%**
- Average relevant retrieved: **1.5-2.0** out of 4.6

### Target (Phase 2-3)
- NDCG@10: **0.30+** (+100%)
- Zero-score queries: **<25%**
- Average relevant retrieved: **2.5-3.0** out of 4.6

### Stretch Goal
- NDCG@10: **0.40+** (+167%)
- Competitive with other datasets
- Zero-score queries: **<15%**

---

## 📚 References

**Analysis Files:**
- `analyze_multiheirtt.py` - Query-level analysis script
- `multiheirtt_query_analysis.csv` - Detailed per-query metrics
- `notebook/3. improved_chunking_pipeline.ipynb` - Current pipeline

**Key Findings Documents:**
- Worst queries: 50% have NDCG = 0
- Best queries: Max NDCG = 0.6367 (still not perfect)
- Average relevant docs: 4.6 per query
- Main issue: **Numerical reasoning** + **Multi-document aggregation**

---

## ✅ Conclusion

MultiHeirTT is the **bottleneck** in overall performance. Current approach gets **0.1497 NDCG@10** while other datasets average **0.50-0.87**.

**Why it matters:**
- MultiHeirTT has 974 queries (21% of total)
- Its low score drags down overall NDCG significantly
- Improving it from 0.15 → 0.30 would boost overall NDCG by ~0.03-0.04

**Next Action:**
1. ✅ **Understand the problem** (Done - this analysis)
2. 🔄 **Test quick wins** (Increase top_k + BM25 weight)
3. 📊 **Measure impact** (Re-run evaluation)
4. 🚀 **Iterate** (Implement Phase 2-3 based on results)

**Time Investment vs Impact:**
- Phase 1: 1-2 hours → +35% improvement
- Phase 2: 1-2 days → +67% improvement
- Phase 3: 3-5 days → +100%+ improvement

**Recommendation:** Start with Phase 1 (high ROI), then decide on Phase 2/3 based on results and time constraints.
