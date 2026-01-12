"""
🔍 MULTIHEIRTT Query-Level Analysis
Phân tích chi tiết từng query để tìm patterns trong best/worst cases
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = Path("../data")
SUBMISSION_FILE = "submission_optimal_chunking.csv"
QRELS_FILE = DATA_DIR / "MultiHeirtt_qrels.tsv"
QUERIES_FILE = DATA_DIR / "multiheirtt_queries.jsonl" / "queries.jsonl"
CORPUS_FILE = DATA_DIR / "multiheirtt_corpus.jsonl" / "corpus.jsonl"

# ============================================================================
# Helper Functions
# ============================================================================

def load_qrels(qrels_path):
    """Load qrels file"""
    qrels_df = pd.read_csv(qrels_path, sep='\t')
    
    # Group by query_id
    qrels = {}
    for _, row in qrels_df.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        score = row['score']
        
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][corpus_id] = score
    
    return qrels, qrels_df


def load_queries(queries_path):
    """Load queries from JSONL"""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    return queries


def load_corpus(corpus_path):
    """Load corpus from JSONL"""
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = {
                'title': doc.get('title', ''),
                'text': doc.get('text', '')
            }
    return corpus


def load_submission(submission_path):
    """Load submission file"""
    sub_df = pd.read_csv(submission_path)
    
    # Group by query_id
    results = {}
    for _, row in sub_df.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        
        if query_id not in results:
            results[query_id] = []
        results[query_id].append(corpus_id)
    
    return results, sub_df


def compute_ndcg_single(relevant_docs, retrieved_docs, k=10):
    """Compute NDCG@k for a single query"""
    retrieved_k = retrieved_docs[:k]
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        rel = relevant_docs.get(doc_id, 0)
        dcg += rel / np.log2(i + 2)
    
    # IDCG
    ideal_scores = sorted(relevant_docs.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def analyze_query_characteristics(query_text):
    """Phân tích đặc điểm của query"""
    words = query_text.lower().split()
    
    # Check for numbers
    has_numbers = any(any(c.isdigit() for c in w) for w in words)
    
    # Check for comparison words
    comparison_words = ['highest', 'lowest', 'largest', 'smallest', 'most', 'least', 
                       'greater', 'less', 'more', 'better', 'worse', 'increase', 'decrease']
    has_comparison = any(cw in query_text.lower() for cw in comparison_words)
    
    # Check for calculation words
    calc_words = ['sum', 'total', 'average', 'mean', 'difference', 'ratio', 'percentage',
                  'calculate', 'compute', 'add', 'subtract', 'multiply', 'divide']
    needs_calculation = any(cw in query_text.lower() for cw in calc_words)
    
    # Check for table-related words
    table_words = ['table', 'row', 'column', 'cell', 'value']
    mentions_table = any(tw in query_text.lower() for tw in table_words)
    
    return {
        'length': len(words),
        'has_numbers': has_numbers,
        'has_comparison': has_comparison,
        'needs_calculation': needs_calculation,
        'mentions_table': mentions_table
    }


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("🔍 MULTIHEIRTT QUERY-LEVEL ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    qrels, qrels_df = load_qrels(QRELS_FILE)
    queries = load_queries(QUERIES_FILE)
    corpus = load_corpus(CORPUS_FILE)
    results, sub_df = load_submission(SUBMISSION_FILE)
    
    # Filter for MULTIHEIRTT queries only
    multiheirtt_query_ids = [qid for qid in qrels.keys() if qid in results]
    
    print(f"   ✅ Loaded {len(qrels)} qrels entries")
    print(f"   ✅ Loaded {len(queries)} queries")
    print(f"   ✅ Loaded {len(corpus)} documents")
    print(f"   ✅ Loaded {len(results)} results")
    print(f"   ✅ Found {len(multiheirtt_query_ids)} MULTIHEIRTT queries in submission")
    
    # Compute NDCG for each query
    print("\n📊 Computing NDCG@10 for each query...")
    query_scores = []
    
    for query_id in multiheirtt_query_ids:
        relevant = qrels[query_id]
        retrieved = results[query_id]
        
        ndcg = compute_ndcg_single(relevant, retrieved, k=10)
        
        query_text = queries.get(query_id, "")
        characteristics = analyze_query_characteristics(query_text)
        
        # Count relevant docs
        num_relevant = len(relevant)
        num_retrieved_relevant = sum(1 for doc_id in retrieved[:10] if doc_id in relevant)
        
        query_scores.append({
            'query_id': query_id,
            'ndcg_10': ndcg,
            'query_text': query_text,
            'num_relevant': num_relevant,
            'num_retrieved_relevant': num_retrieved_relevant,
            **characteristics
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(query_scores)
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Mean NDCG@10: {df['ndcg_10'].mean():.4f}")
    print(f"   Median NDCG@10: {df['ndcg_10'].median():.4f}")
    print(f"   Std NDCG@10: {df['ndcg_10'].std():.4f}")
    print(f"   Min NDCG@10: {df['ndcg_10'].min():.4f}")
    print(f"   Max NDCG@10: {df['ndcg_10'].max():.4f}")
    
    # Distribution
    print(f"\n📊 NDCG Distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(df['ndcg_10'], bins=bins)
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        pct = (hist[i] / len(df)) * 100
        print(f"   [{low:.1f}-{high:.1f}): {hist[i]:4d} queries ({pct:5.1f}%)")
    
    # Perfect vs Zero scores
    perfect = (df['ndcg_10'] == 1.0).sum()
    zero = (df['ndcg_10'] == 0.0).sum()
    print(f"\n   Perfect scores (1.0): {perfect} ({perfect/len(df)*100:.1f}%)")
    print(f"   Zero scores (0.0): {zero} ({zero/len(df)*100:.1f}%)")
    
    # ========================================================================
    # WORST CASES
    # ========================================================================
    print("\n" + "="*80)
    print("❌ WORST 10 QUERIES (Lowest NDCG@10)")
    print("="*80)
    
    worst = df.nsmallest(10, 'ndcg_10')
    
    for idx, row in worst.iterrows():
        print(f"\n🔴 Query ID: {row['query_id']}")
        print(f"   NDCG@10: {row['ndcg_10']:.4f}")
        print(f"   Relevant docs: {row['num_relevant']}")
        print(f"   Retrieved relevant (top-10): {row['num_retrieved_relevant']}")
        print(f"   Query: {row['query_text'][:150]}...")
        print(f"   Characteristics:")
        print(f"      - Length: {row['length']} words")
        print(f"      - Has numbers: {row['has_numbers']}")
        print(f"      - Has comparison: {row['has_comparison']}")
        print(f"      - Needs calculation: {row['needs_calculation']}")
        print(f"      - Mentions table: {row['mentions_table']}")
        
        # Show what was retrieved
        retrieved = results[row['query_id']][:5]
        relevant_ids = qrels[row['query_id']]
        print(f"   Top-5 retrieved:")
        for i, doc_id in enumerate(retrieved, 1):
            is_relevant = "✅" if doc_id in relevant_ids else "❌"
            title = corpus.get(doc_id, {}).get('title', 'N/A')[:60]
            print(f"      {i}. {is_relevant} {doc_id}: {title}")
    
    # ========================================================================
    # BEST CASES
    # ========================================================================
    print("\n" + "="*80)
    print("✅ BEST 10 QUERIES (Highest NDCG@10)")
    print("="*80)
    
    best = df.nlargest(10, 'ndcg_10')
    
    for idx, row in best.iterrows():
        print(f"\n🟢 Query ID: {row['query_id']}")
        print(f"   NDCG@10: {row['ndcg_10']:.4f}")
        print(f"   Relevant docs: {row['num_relevant']}")
        print(f"   Retrieved relevant (top-10): {row['num_retrieved_relevant']}")
        print(f"   Query: {row['query_text'][:150]}...")
        print(f"   Characteristics:")
        print(f"      - Length: {row['length']} words")
        print(f"      - Has numbers: {row['has_numbers']}")
        print(f"      - Has comparison: {row['has_comparison']}")
        print(f"      - Needs calculation: {row['needs_calculation']}")
        print(f"      - Mentions table: {row['mentions_table']}")
    
    # ========================================================================
    # PATTERN ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("🔍 PATTERN ANALYSIS")
    print("="*80)
    
    # Split into good vs bad
    threshold = df['ndcg_10'].median()
    good_queries = df[df['ndcg_10'] >= threshold]
    bad_queries = df[df['ndcg_10'] < threshold]
    
    print(f"\nUsing median ({threshold:.4f}) as threshold:")
    print(f"   Good queries (>= median): {len(good_queries)}")
    print(f"   Bad queries (< median): {len(bad_queries)}")
    
    # Compare characteristics
    print(f"\n📊 Characteristics Comparison:")
    print(f"{'Characteristic':<25} {'Good Queries':<15} {'Bad Queries':<15} {'Difference'}")
    print("-"*75)
    
    for col in ['length', 'has_numbers', 'has_comparison', 'needs_calculation', 'mentions_table']:
        good_val = good_queries[col].mean()
        bad_val = bad_queries[col].mean()
        diff = good_val - bad_val
        
        if col == 'length':
            print(f"{col:<25} {good_val:<15.1f} {bad_val:<15.1f} {diff:+.1f}")
        else:
            print(f"{col:<25} {good_val:<15.1%} {bad_val:<15.1%} {diff:+.1%}")
    
    # Relevant docs analysis
    print(f"\n📚 Relevant Documents:")
    print(f"   Good queries - avg relevant: {good_queries['num_relevant'].mean():.1f}")
    print(f"   Bad queries - avg relevant: {bad_queries['num_relevant'].mean():.1f}")
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    # Analyze patterns
    recs = []
    
    if bad_queries['has_numbers'].mean() > good_queries['has_numbers'].mean():
        recs.append("⚠️ Queries with numbers perform WORSE")
        recs.append("   → Consider increasing BM25 weight further (currently 60%)")
        recs.append("   → Add number-aware preprocessing (normalize numbers)")
    
    if bad_queries['needs_calculation'].mean() > good_queries['needs_calculation'].mean():
        recs.append("⚠️ Queries needing calculations perform WORSE")
        recs.append("   → These queries need multi-document reasoning")
        recs.append("   → Consider retrieving more documents (increase top_k)")
    
    if bad_queries['length'].mean() > good_queries['length'].mean():
        recs.append("⚠️ Longer queries perform WORSE")
        recs.append("   → Consider query reformulation or key phrase extraction")
    
    if bad_queries['num_relevant'].mean() > good_queries['num_relevant'].mean():
        recs.append("⚠️ Queries with more relevant docs perform WORSE")
        recs.append("   → Need better ranking to surface correct docs")
        recs.append("   → Consider ensemble reranking or LLM-based reranking")
    
    if zero > len(df) * 0.3:
        recs.append(f"⚠️ High proportion of zero scores ({zero/len(df)*100:.1f}%)")
        recs.append("   → Retrieval is missing relevant documents entirely")
        recs.append("   → Consider query expansion or semantic search improvements")
    
    if not recs:
        recs.append("✅ No clear pattern identified - need deeper investigation")
    
    for rec in recs:
        print(rec)
    
    # Save detailed results
    output_file = "multiheirtt_query_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
