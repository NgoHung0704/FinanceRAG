
# Integration code for your retrieval pipeline
# Add this to your financerag/retrieval/dense.py or similar

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# Load chunked corpus
chunked_corpus = []
with open('..\data\chunked_corpus\financebench_corpus_chunked_production.jsonl', 'r') as f:
    for line in f:
        chunked_corpus.append(json.loads(line))

# Initialize model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Encode corpus
corpus_texts = [doc['text'] for doc in chunked_corpus]
corpus_embeddings = model.encode(corpus_texts, show_progress_bar=True)

# Retrieval function
def retrieve(query, top_k=10):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'doc_id': chunked_corpus[idx]['_id'],
            'score': scores[idx],
            'text': chunked_corpus[idx]['text'],
            'original_id': chunked_corpus[idx].get('original_id', chunked_corpus[idx]['_id'])
        })
    return results

# Test
results = retrieve("What is the company's revenue?")
for i, r in enumerate(results, 1):
    print(f"{i}. Score: {r['score']:.4f}")
    print(f"   {r['text'][:100]}...\n")
