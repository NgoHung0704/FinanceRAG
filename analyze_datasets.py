import json
import statistics
import os

base_path = r'd:\Documents\INSA_Lyon\INSA 4A\TCD 1\FinanceRAG\data'
datasets = ['multiheirtt', 'tatqa', 'convfinqa', 'financebench', 'finqabench', 'finder', 'finqa']

print("="*80)
print("DATASET ANALYSIS FOR FINANCERAG")
print("="*80)

for ds in datasets:
    corpus_path = os.path.join(base_path, f"{ds}_corpus.jsonl", "corpus.jsonl")
    queries_path = os.path.join(base_path, f"{ds}_queries.jsonl", "queries.jsonl")
    
    # Count total
    with open(corpus_path, encoding='utf-8') as f:
        corpus_docs = [json.loads(l) for l in f.readlines()]
    
    with open(queries_path, encoding='utf-8') as f:
        queries = [json.loads(l) for l in f.readlines()]
    
    # Analyze first 500 docs
    sample = corpus_docs[:min(500, len(corpus_docs))]
    
    doc_lens = [len(d['text']) for d in sample]
    query_lens = [len(q['text']) for q in queries[:min(200, len(queries))]]
    
    has_tables = sum(1 for d in sample if '|' in d['text'] or '\t' in d['text'])
    has_long_docs = sum(1 for l in doc_lens if l > 2000)
    has_short_docs = sum(1 for l in doc_lens if l < 500)
    
    # Check for numbers
    has_numbers = sum(1 for d in sample if any(c.isdigit() for c in d['text']))
    
    print(f"\n{ds.upper()}:")
    print(f"  📊 Corpus size: {len(corpus_docs):,}")
    print(f"  📋 Query size: {len(queries):,}")
    print(f"  📏 Avg doc length: {statistics.mean(doc_lens):.0f} chars")
    print(f"  📐 Median doc length: {statistics.median(doc_lens):.0f} chars")
    print(f"  📈 Max doc length: {max(doc_lens):,} chars")
    print(f"  📉 Min doc length: {min(doc_lens)} chars")
    print(f"  📊 Tables detected: {has_tables}/{len(sample)} ({has_tables/len(sample)*100:.1f}%)")
    print(f"  🔢 Numeric content: {has_numbers}/{len(sample)} ({has_numbers/len(sample)*100:.1f}%)")
    print(f"  📄 Long docs (>2000): {has_long_docs} ({has_long_docs/len(sample)*100:.1f}%)")
    print(f"  📄 Short docs (<500): {has_short_docs} ({has_short_docs/len(sample)*100:.1f}%)")
    print(f"  ❓ Avg query length: {statistics.mean(query_lens):.0f} chars")
    
    # Sample queries
    print(f"  🔍 Sample queries:")
    for i, q in enumerate(queries[:3]):
        print(f"     {i+1}. {q['text'][:80]}...")

print("\n" + "="*80)
