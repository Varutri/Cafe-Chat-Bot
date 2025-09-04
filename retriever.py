# retriever.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

class MyRetriever:
    """Simple wrapper retriever for FAISS indexer"""
    def __init__(self, faiss_indexer):
        self.indexer = faiss_indexer


class HybridRetriever:
    """
    Hybrid retriever combining:
      - Dense embeddings (via FAISS indexer)
      - Sparse TF-IDF recall
    """
    def __init__(self, docs: List[Dict[str, Any]], embedder, faiss_indexer, max_features: int = 5000):
        self.docs = docs
        self.texts = [d['text'] for d in docs]
        self.embedder = embedder
        self.indexer = faiss_indexer

        # Fit TF-IDF on corpus (1-2 grams)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def dense_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search using dense embeddings + FAISS"""
        q_emb = self.embedder.encode([query])  # already normalized
        results = self.indexer.search(q_emb, top_k=k)
        return results

    def sparse_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search using TF-IDF cosine similarity"""
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().ravel()

        top_idx = np.argsort(-scores)[:k]
        results = [
            {"index": i, "meta": self.docs[i]['meta'], "score": float(scores[i])}
            for i in top_idx if scores[i] > 0
        ]
        return results

    def hybrid_search(self, query: str, k: int = 8, alpha: float = 0.6) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense + sparse scores.
        Args:
            query: user query string
            k: number of results to return
            alpha: weight for dense (0-1). Higher alpha → more dense influence.
        """
        dense = self.dense_search(query, k=k * 2)
        sparse = self.sparse_search(query, k=k * 2)

        combined = {}

        # Add dense results
        for r in dense:
            idx = r['index']
            combined[idx] = {
                "meta": r['meta'],
                "dense_score": r['score'],
                "sparse_score": 0.0
            }

        # Merge sparse results
        for r in sparse:
            idx = r['index']
            if idx in combined:
                combined[idx]["sparse_score"] = r['score']
            else:
                combined[idx] = {
                    "meta": r['meta'],
                    "dense_score": 0.0,
                    "sparse_score": r['score']
                }

        # Re-rank by weighted score
        results = [
            {
                "index": idx,
                "meta": v['meta'],
                "score": alpha * v['dense_score'] + (1 - alpha) * v['sparse_score']
            }
            for idx, v in combined.items()
        ]

        return sorted(results, key=lambda x: -x['score'])[:k]
