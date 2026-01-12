"""
FinanceRAG Shared Utilities
===========================

Module này chứa các helper functions dùng chung cho tất cả notebooks.
Import bằng cách thêm sys.path và import:

    import sys
    sys.path.insert(0, '..')
    from utils import load_jsonl, compute_ndcg, ...

Hoặc nếu đang ở trong subdirectory:
    
    import sys
    sys.path.insert(0, '../..')
    from notebook.utils import load_jsonl, compute_ndcg, ...
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm.auto import tqdm


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load a JSONL file and return list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, each representing a line in the file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_jsonl_data(dataset_name: str, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load corpus, queries, and qrels for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'convfinqa', 'tatqa')
        data_dir: Base directory containing data folders
        
    Returns:
        Tuple of (corpus_df, queries_df, qrels_df)
        qrels_df may be None if not found
    """
    corpus_path = os.path.join(data_dir, f"{dataset_name}_corpus.jsonl", "corpus.jsonl")
    queries_path = os.path.join(data_dir, f"{dataset_name}_queries.jsonl", "queries.jsonl")
    
    # Map dataset names to qrels files
    qrels_mapping = {
        'convfinqa': 'ConvFinQA_qrels.tsv',
        'financebench': 'FinanceBench_qrels.tsv',
        'finder': 'FinDER_qrels.tsv',
        'finqa': 'FinQA_qrels.tsv',
        'finqabench': 'FinQABench_qrels.tsv',
        'multiheirtt': 'MultiHeirtt_qrels.tsv',
        'tatqa': 'TATQA_qrels.tsv'
    }
    
    qrels_path = os.path.join(data_dir, qrels_mapping.get(dataset_name, f"{dataset_name}_qrels.tsv"))
    
    corpus_df = pd.read_json(corpus_path, lines=True)
    queries_df = pd.read_json(queries_path, lines=True)
    
    qrels_df = None
    if os.path.exists(qrels_path):
        qrels_df = pd.read_csv(qrels_path, sep='\t')
    
    print(f"  Loaded {len(corpus_df)} docs, {len(queries_df)} queries")
    return corpus_df, queries_df, qrels_df


def load_dataset_for_evaluation(
    dataset_name: str, 
    data_dir: Union[str, Path]
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    """
    Load dataset for evaluation: corpus, queries, and qrels as dict.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Base directory containing data folders
        
    Returns:
        Tuple of (corpus, queries, qrels_dict)
        - corpus: List of {'_id': ..., 'text': ..., 'title': ...}
        - queries: List of {'_id': ..., 'text': ...}
        - qrels_dict: Dict[query_id, Dict[doc_id, relevance_score]]
    """
    data_dir = Path(data_dir)
    
    # Load corpus
    corpus_path = data_dir / f"{dataset_name}_corpus.jsonl" / "corpus.jsonl"
    corpus = load_jsonl(corpus_path)
    
    # Load queries
    queries_path = data_dir / f"{dataset_name}_queries.jsonl" / "queries.jsonl"
    queries = load_jsonl(queries_path)
    
    # Load qrels
    qrels_mapping = {
        'convfinqa': 'ConvFinQA_qrels.tsv',
        'financebench': 'FinanceBench_qrels.tsv',
        'finder': 'FinDER_qrels.tsv',
        'finqa': 'FinQA_qrels.tsv',
        'finqabench': 'FinQABench_qrels.tsv',
        'multiheirtt': 'MultiHeirtt_qrels.tsv',
        'tatqa': 'TATQA_qrels.tsv'
    }
    
    qrels_path = data_dir / qrels_mapping.get(dataset_name, f"{dataset_name}_qrels.tsv")
    
    qrels_dict = {}
    if qrels_path.exists():
        qrels_df = pd.read_csv(qrels_path, sep='\t')
        
        # Handle both column naming conventions
        query_col = 'query_id' if 'query_id' in qrels_df.columns else 'query-id'
        corpus_col = 'corpus_id' if 'corpus_id' in qrels_df.columns else 'corpus-id'
        
        for _, row in qrels_df.iterrows():
            qid = str(row[query_col])
            did = str(row[corpus_col])
            score = int(row['score'])
            
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][did] = score
    
    print(f"  Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels_dict)} qrels")
    return corpus, queries, qrels_dict


def load_prechunked_corpus(
    dataset_name: str, 
    chunked_dir: Union[str, Path], 
    config_file: Optional[str] = None
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Load pre-chunked corpus from chunking evaluation.
    
    Args:
        dataset_name: Name of the dataset
        chunked_dir: Directory containing chunked corpus files
        config_file: Optional path to chunking config JSON file
        
    Returns:
        Tuple of (chunks, chunking_method_description)
        Returns (None, None) if file not found
    """
    chunked_path = Path(chunked_dir) / f"{dataset_name}_corpus_chunked_optimal.jsonl"
    
    if not chunked_path.exists():
        print(f"  ⚠️ Pre-chunked file not found: {chunked_path}")
        return None, None
    
    # Load chunking config to show which method was used
    chunking_method = "unknown"
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            configs = json.load(f)
            if dataset_name in configs:
                cfg = configs[dataset_name]
                method = cfg.get('method', 'unknown')
                if method != 'no_chunking' and cfg.get('chunk_size'):
                    chunking_method = f"{method} ({cfg['chunk_size']}/{cfg['chunk_overlap']})"
                else:
                    chunking_method = method
    
    # Load pre-chunked corpus
    chunks = load_jsonl(chunked_path)
    
    print(f"  ✅ Loaded {len(chunks)} pre-chunked chunks")
    print(f"  📋 Method used: {chunking_method}")
    return chunks, chunking_method


# =============================================================================
# NDCG AND EVALUATION METRICS
# =============================================================================

def compute_ndcg(
    retrieved_docs: List[str], 
    relevance_scores: Dict[str, int], 
    k: int = 10,
    handle_chunks: bool = False
) -> float:
    """
    Compute NDCG@k score (Normalized Discounted Cumulative Gain).
    
    This is the STANDARDIZED implementation for all FinanceRAG notebooks.
    Uses the formula: DCG = Σ (2^rel - 1) / log2(i + 1)
    
    Args:
        retrieved_docs: List of retrieved document IDs (ranked by score, descending)
        relevance_scores: Dict mapping doc_id to relevance score (typically 0 or 1)
        k: Number of top results to consider
        handle_chunks: If True, handles chunked doc IDs (e.g., "doc_chunk_0")
                      by extracting original doc ID and deduplicating
        
    Returns:
        NDCG@k score in range [0.0, 1.0]
        
    Example:
        >>> retrieved = ['doc1', 'doc2', 'doc3']
        >>> qrels = {'doc1': 1, 'doc3': 1}  # doc1 and doc3 are relevant
        >>> compute_ndcg(retrieved, qrels, k=10)
        0.8154...
    """
    if not relevance_scores:
        return 0.0
    
    # Handle chunked document IDs if needed
    if handle_chunks:
        seen_original_ids = set()
        gains = []
        
        for doc_id in retrieved_docs:
            if len(gains) >= k:
                break
                
            # Extract original doc ID from chunk ID
            original_id = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
            
            # Skip if we've already counted this original doc
            if original_id in seen_original_ids:
                continue
            seen_original_ids.add(original_id)
            
            rel = relevance_scores.get(original_id, 0)
            gains.append(rel)
    else:
        # Standard case: no chunk handling
        retrieved_k = retrieved_docs[:k]
        gains = [relevance_scores.get(doc_id, 0) for doc_id in retrieved_k]
    
    # Pad to k if needed (remaining positions get 0 gain)
    while len(gains) < k:
        gains.append(0)
    
    # Compute DCG: sum of (2^rel - 1) / log2(rank + 1)
    # Note: rank starts at 1, so denominator is log2(i + 2) for 0-indexed
    dcg = 0.0
    for i, gain in enumerate(gains[:k]):
        if gain > 0:
            dcg += (2 ** gain - 1) / np.log2(i + 2)
    
    # Compute IDCG (Ideal DCG): best possible ranking
    ideal_gains = sorted(
        [s for s in relevance_scores.values() if s > 0], 
        reverse=True
    )[:k]
    
    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        idcg += (2 ** gain - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return float(min(max(ndcg, 0.0), 1.0))


def compute_ndcg_batch(
    qrels: Dict[str, Dict[str, int]], 
    results: Dict[str, List[str]], 
    k: int = 10
) -> float:
    """
    Compute average NDCG@k across multiple queries.
    
    Args:
        qrels: Dict mapping query_id to {doc_id: relevance_score}
        results: Dict mapping query_id to list of retrieved doc_ids (ranked)
        k: Number of top results to consider
        
    Returns:
        Mean NDCG@k score across all queries
    """
    ndcg_scores = []
    
    for query_id, retrieved in results.items():
        if query_id not in qrels:
            continue
        
        relevance = qrels[query_id]
        ndcg = compute_ndcg(retrieved, relevance, k)
        ndcg_scores.append(ndcg)
    
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_recall_at_k(
    retrieved_docs: List[str], 
    relevance_scores: Dict[str, int], 
    k: int = 10
) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score
        k: Number of top results to consider
        
    Returns:
        Recall@k score in range [0.0, 1.0]
    """
    relevant_docs = set(doc_id for doc_id, score in relevance_scores.items() if score > 0)
    if not relevant_docs:
        return 0.0
    
    retrieved_relevant = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
    return float(retrieved_relevant / len(relevant_docs))


def compute_mrr(
    retrieved_docs: List[str], 
    relevance_scores: Dict[str, int]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score
        
    Returns:
        MRR score in range [0.0, 1.0]
    """
    relevant_docs = set(doc_id for doc_id, score in relevance_scores.items() if score > 0)
    
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_docs:
            return 1.0 / i
    
    return 0.0


def evaluate_results_df(
    results_df: pd.DataFrame, 
    qrels_df: pd.DataFrame, 
    k: int = 10
) -> Dict[str, Any]:
    """
    Evaluate results DataFrame against qrels.
    
    Args:
        results_df: DataFrame with columns ['query_id', 'corpus_id', 'score']
        qrels_df: DataFrame with columns ['query_id'/'query-id', 'corpus_id'/'corpus-id', 'score']
        k: Number of top results to consider
        
    Returns:
        Dict with 'NDCG@10', 'num_queries', 'num_qrels'
    """
    # Handle both column naming conventions in qrels
    query_col = 'query_id' if 'query_id' in qrels_df.columns else 'query-id'
    corpus_col = 'corpus_id' if 'corpus_id' in qrels_df.columns else 'corpus-id'
    
    # Convert qrels to dict format
    qrels = {}
    for _, row in qrels_df.iterrows():
        qid = str(row[query_col])
        did = str(row[corpus_col])
        score = int(row['score'])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = score
    
    # Convert results to dict format
    results = results_df.groupby('query_id')['corpus_id'].apply(list).to_dict()
    
    # Compute NDCG
    ndcg = compute_ndcg_batch(qrels, results, k)
    
    return {
        'NDCG@10': ndcg, 
        'num_queries': len(results), 
        'num_qrels': len(qrels)
    }


# =============================================================================
# HYBRID RETRIEVAL UTILITIES
# =============================================================================

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        scores: Array of scores
        
    Returns:
        Normalized scores in [0, 1]
    """
    if len(scores) == 0:
        return scores
    
    score_min = scores.min()
    score_max = scores.max()
    
    if score_max == score_min:
        return np.ones_like(scores)
    
    return (scores - score_min) / (score_max - score_min)


def hybrid_search(
    query_emb: np.ndarray,
    query_text: str,
    faiss_index,
    bm25,
    corpus_texts: List[str],
    top_k: int,
    alpha: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid search combining dense retrieval and BM25.
    
    Args:
        query_emb: Query embedding vector
        query_text: Query text for BM25
        faiss_index: FAISS index for dense retrieval
        bm25: BM25Okapi model
        corpus_texts: List of corpus texts (for BM25)
        top_k: Number of results to return
        alpha: Weight for dense scores (1-alpha for BM25)
               Higher alpha = more weight on dense retrieval
               
    Returns:
        Tuple of (hybrid_scores, indices)
    """
    # Dense retrieval (retrieve more candidates for re-scoring)
    dense_scores, indices = faiss_index.search(
        query_emb.reshape(1, -1).astype('float32'), 
        top_k * 2
    )
    dense_scores = dense_scores[0]
    indices = indices[0]
    
    # BM25 retrieval
    query_tokens = query_text.lower().split()
    bm25_all_scores = bm25.get_scores(query_tokens)
    bm25_scores = bm25_all_scores[indices]
    
    # Normalize both score sets
    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)
    
    # Combine with alpha weighting
    hybrid_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    
    # Re-sort by hybrid scores and take top_k
    sorted_idx = np.argsort(hybrid_scores)[::-1][:top_k]
    
    return hybrid_scores[sorted_idx], indices[sorted_idx]


def aggregate_chunk_scores(
    chunk_scores: Dict[str, List[float]], 
    method: str = 'max'
) -> Dict[str, float]:
    """
    Aggregate chunk-level scores to document-level scores.
    
    Args:
        chunk_scores: Dict mapping doc_id to list of chunk scores
        method: Aggregation method - 'max', 'mean', or 'weighted'
        
    Returns:
        Dict mapping doc_id to aggregated score
    """
    aggregated = {}
    
    for doc_id, scores in chunk_scores.items():
        if not scores:
            aggregated[doc_id] = 0.0
            continue
            
        if method == 'max':
            aggregated[doc_id] = max(scores)
        elif method == 'mean':
            aggregated[doc_id] = float(np.mean(scores))
        elif method == 'weighted':
            # Higher weight for earlier (higher-ranked) chunks
            weights = np.array([1.0 / (i + 1) for i in range(len(scores))])
            weights = weights / weights.sum()
            aggregated[doc_id] = float(np.dot(scores, weights))
        else:
            # Default to max
            aggregated[doc_id] = max(scores)
    
    return aggregated


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def sample_queries(
    queries: List[Dict], 
    qrels: Dict[str, Dict], 
    n_queries: int = 30,
    seed: int = 42
) -> List[Dict]:
    """
    Sample queries that have relevance judgments.
    
    Args:
        queries: List of query dicts with '_id' field
        qrels: Dict mapping query_id to relevance dict
        n_queries: Number of queries to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled query dicts
    """
    # Filter queries that have qrels
    valid_queries = [q for q in queries if q['_id'] in qrels]
    
    if len(valid_queries) == 0:
        return []
    
    # Sample
    n_sample = min(n_queries, len(valid_queries))
    np.random.seed(seed)
    sampled = list(np.random.choice(valid_queries, n_sample, replace=False))
    
    return sampled


def smart_sample_corpus(
    corpus: List[Dict],
    qrels: Dict[str, Dict[str, int]],
    queries_sample: List[Dict],
    max_size: int = 500,
    seed: int = 42
) -> List[Dict]:
    """
    Smart corpus sampling that ensures relevant documents are included.
    
    Args:
        corpus: Full corpus list
        qrels: Relevance judgments
        queries_sample: Sampled queries
        max_size: Maximum corpus size
        seed: Random seed
        
    Returns:
        Sampled corpus ensuring relevant docs are included
    """
    # Get all relevant doc IDs from sampled queries
    relevant_ids = set()
    for q in queries_sample:
        qid = q['_id']
        if qid in qrels:
            relevant_ids.update(qrels[qid].keys())
    
    # Create lookup and collect relevant docs
    corpus_lookup = {doc['_id']: doc for doc in corpus}
    
    sampled = []
    for doc_id in relevant_ids:
        if doc_id in corpus_lookup:
            sampled.append(corpus_lookup[doc_id])
    
    # Fill remaining slots with random non-relevant docs
    remaining_space = max_size - len(sampled)
    if remaining_space > 0:
        sampled_ids = set(d['_id'] for d in sampled)
        non_relevant = [doc for doc in corpus if doc['_id'] not in sampled_ids]
        
        np.random.seed(seed)
        if len(non_relevant) > remaining_space:
            additional_idx = np.random.choice(
                len(non_relevant), 
                remaining_space, 
                replace=False
            )
            additional = [non_relevant[i] for i in additional_idx]
        else:
            additional = non_relevant
        
        sampled.extend(additional)
    
    return sampled


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def prepare_corpus_texts(
    corpus_df: pd.DataFrame, 
    combine_title: bool = True,
    max_length: int = 8192
) -> List[str]:
    """
    Prepare corpus texts for embedding.
    
    Args:
        corpus_df: DataFrame with 'text' and optionally 'title' columns
        combine_title: Whether to prepend title to text
        max_length: Maximum text length in characters
        
    Returns:
        List of prepared text strings
    """
    texts = []
    for _, row in corpus_df.iterrows():
        text = str(row.get('text', ''))
        title = str(row.get('title', ''))
        
        if combine_title and title:
            full_text = f"[{title}] {text}"
        else:
            full_text = text
        
        texts.append(full_text[:max_length])
    
    return texts


def prepare_query_for_model(query: str, model_name: str) -> str:
    """
    Prepare query text for specific embedding models.
    
    Some models (E5, BGE) may need special prefixes.
    
    Args:
        query: Query text
        model_name: Name of the embedding model
        
    Returns:
        Prepared query text
    """
    model_lower = model_name.lower()
    
    # E5 models need "query: " prefix
    if 'e5' in model_lower:
        return f"query: {query}"
    
    # Intfloat models may need instruction
    if 'intfloat' in model_lower and 'e5' in model_lower:
        return f"query: {query}"
    
    return query


def prepare_document_for_model(doc_text: str, model_name: str) -> str:
    """
    Prepare document text for specific embedding models.
    
    Args:
        doc_text: Document text
        model_name: Name of the embedding model
        
    Returns:
        Prepared document text
    """
    model_lower = model_name.lower()
    
    # E5 models need "passage: " prefix for documents
    if 'e5' in model_lower:
        return f"passage: {doc_text}"
    
    return doc_text


# =============================================================================
# DATASET CONSTANTS
# =============================================================================

DATASETS = [
    'convfinqa',
    'financebench', 
    'finder',
    'finqa',
    'finqabench',
    'multiheirtt',
    'tatqa'
]

QRELS_MAPPING = {
    'convfinqa': 'ConvFinQA_qrels.tsv',
    'financebench': 'FinanceBench_qrels.tsv',
    'finder': 'FinDER_qrels.tsv',
    'finqa': 'FinQA_qrels.tsv',
    'finqabench': 'FinQABench_qrels.tsv',
    'multiheirtt': 'MultiHeirtt_qrels.tsv',
    'tatqa': 'TATQA_qrels.tsv'
}


# =============================================================================
# PRINT UTILITIES
# =============================================================================

def print_evaluation_summary(eval_results: Dict[str, Dict], baseline: float = 0.328):
    """
    Print formatted evaluation summary.
    
    Args:
        eval_results: Dict mapping dataset_name to {'NDCG@10': ..., 'num_queries': ...}
        baseline: Baseline NDCG score for comparison
    """
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY (NDCG@10)")
    print("="*70)
    
    total_ndcg = 0
    total_queries = 0
    
    for ds, m in sorted(eval_results.items()):
        print(f"\n{ds.upper():15s}: {m['NDCG@10']:.4f} ({m.get('num_queries', '?')} queries)")
        if 'num_qrels' in m:
            total_ndcg += m['NDCG@10'] * m['num_qrels']
            total_queries += m['num_qrels']
    
    if total_queries > 0:
        avg_ndcg = total_ndcg / total_queries
        print(f"\n{'='*70}")
        print(f"📈 WEIGHTED AVERAGE NDCG@10: {avg_ndcg:.4f}")
        print(f"{'='*70}")
        
        gain = avg_ndcg - baseline
        gain_pct = (gain / baseline) * 100
        
        print(f"\n🎯 Performance vs Baseline:")
        print(f"   Baseline: {baseline:.4f}")
        print(f"   Current: {avg_ndcg:.4f}")
        print(f"   Improvement: {'+' if gain >= 0 else ''}{gain:.4f} ({'+' if gain_pct >= 0 else ''}{gain_pct:.1f}%)")


if __name__ == "__main__":
    # Simple test
    print("FinanceRAG Utils Module")
    print("=" * 40)
    print(f"Available datasets: {DATASETS}")
    
    # Test NDCG
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    qrels = {'doc1': 1, 'doc3': 1, 'doc5': 1}
    ndcg = compute_ndcg(retrieved, qrels, k=5)
    print(f"\nTest NDCG@5: {ndcg:.4f}")
