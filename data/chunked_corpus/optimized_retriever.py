"""
Optimized Chunked Retriever for FinanceRAG
Auto-generated from notebook analysis
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import numpy as np

class OptimizedChunkedRetriever:
    """
    Production-ready chunked retriever with configurable strategies
    Based on extensive grid search analysis
    """

    # Optimal configurations from grid search
    CONFIGS = {
        'max_performance': {
            'method': 'recursive',
            'chunk_size': 1000,
            'overlap': 100,
            'description': 'Highest Recall (88.9%), NDCG=92.6%, Expansion=2.04x'
        },
        'balanced': {
            'method': 'character',
            'chunk_size': 1500,
            'overlap': 300,
            'description': 'Best efficiency (85.2% Recall, NDCG=92.6%, Expansion=1.52x) ⭐ RECOMMENDED'
        },
        'memory_efficient': {
            'method': 'character',
            'chunk_size': 1500,
            'overlap': 0,
            'description': 'Lowest memory (81.5% Recall, NDCG=88.9%, Expansion=1.46x)'
        },
        'high_precision': {
            'method': 'recursive',
            'chunk_size': 512,
            'overlap': 51,
            'description': 'Small chunks for precise matching (87% Recall, NDCG=92.6%)'
        }
    }

    def __init__(self, corpus, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 strategy='balanced', aggregation='max'):
        """
        Initialize retriever

        Args:
            corpus: List of documents with '_id' and 'text'
            model_name: Sentence transformer model
            strategy: 'max_performance' | 'balanced' | 'memory_efficient' | 'high_precision'
            aggregation: 'max' | 'mean' | 'weighted'
        """
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.strategy = strategy
        self.aggregation = aggregation

        config = self.CONFIGS.get(strategy, self.CONFIGS['balanced'])
        self.config = config

        print(f"Initializing with strategy: {strategy}")
        print(f"Config: {config['description']}")

        # Create and encode chunks
        self.chunked_corpus = self._create_chunks(corpus, config)
        self._build_mappings()
        self._encode_chunks()

    def _create_chunks(self, corpus, config):
        """Create chunks using configuration"""
        method = config['method']
        chunk_size = config['chunk_size']
        overlap = config['overlap']

        if method == 'recursive':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )

        chunked = []
        for doc in corpus:
            chunks = splitter.split_text(doc['text'])
            for i, chunk in enumerate(chunks):
                chunked.append({
                    '_id': f"{doc['_id']}_chunk_{i}",
                    'text': chunk,
                    'original_id': doc['_id'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })

        return chunked

    def _build_mappings(self):
        """Build chunk-to-document mappings"""
        self.chunk_to_original = {}
        self.original_to_chunks = {}

        for i, chunk_doc in enumerate(self.chunked_corpus):
            orig_id = chunk_doc.get('original_id', chunk_doc['_id'])
            self.chunk_to_original[i] = orig_id

            if orig_id not in self.original_to_chunks:
                self.original_to_chunks[orig_id] = []
            self.original_to_chunks[orig_id].append(i)

    def _encode_chunks(self):
        """Encode all chunks"""
        print(f"Encoding {len(self.chunked_corpus)} chunks...")
        self.chunk_texts = [doc['text'] for doc in self.chunked_corpus]
        self.chunk_embeddings = self.model.encode(
            self.chunk_texts, 
            show_progress_bar=True, 
            batch_size=32
        )
        print("Encoding complete!")

    def retrieve(self, query, top_k=10, return_chunks=False):
        """
        Retrieve relevant documents

        Args:
            query: Query text
            top_k: Number of results
            return_chunks: Return chunks or aggregated documents

        Returns:
            List of results with scores
        """
        query_embedding = self.model.encode([query])
        chunk_scores = cosine_similarity(query_embedding, self.chunk_embeddings)[0]

        if return_chunks:
            top_indices = np.argsort(chunk_scores)[-top_k:][::-1]
            return [
                {
                    'doc_id': self.chunked_corpus[idx]['_id'],
                    'score': chunk_scores[idx],
                    'text': self.chunked_corpus[idx]['text'],
                    'original_id': self.chunk_to_original[idx]
                }
                for idx in top_indices
            ]

        # Aggregate by document
        doc_scores = {}
        for i, score in enumerate(chunk_scores):
            orig_id = self.chunk_to_original[i]
            if orig_id not in doc_scores:
                doc_scores[orig_id] = []
            doc_scores[orig_id].append(score)

        # Apply aggregation
        final_scores = {}
        for doc_id, scores in doc_scores.items():
            if self.aggregation == 'max':
                final_scores[doc_id] = max(scores)
            elif self.aggregation == 'mean':
                final_scores[doc_id] = np.mean(scores)
            elif self.aggregation == 'weighted':
                weights = [1.0 / (i + 1) for i in range(len(scores))]
                final_scores[doc_id] = np.average(scores, weights=weights)

        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            orig_doc = next((d for d in self.corpus if d['_id'] == doc_id), None)
            results.append({
                'doc_id': doc_id,
                'score': score,
                'text': orig_doc['text'] if orig_doc else ''
            })

        return results
