# src/recommender.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import util


class SimpleRecommender:
    """
    A hybrid recommender that combines:
      - Semantic similarity (sentence embeddings)
      - Popularity (# of orders)
      - Ratings (avg user rating)
    """

    def __init__(self, docs, embedder):
        """
        Args:
            docs: list of dicts with 'text' and 'meta' (must include 'num_orders' and 'avg_rating')
            embedder: a sentence-transformers model wrapper
        """
        self.docs = docs
        self.texts = [d['text'] for d in docs]
        self.embedder = embedder

        # Extract metadata
        self.pop = np.array([d['meta'].get('num_orders', 0) for d in docs], dtype=float)
        self.rating = np.array([d['meta'].get('avg_rating', 0.0) for d in docs], dtype=float)

        # Normalize popularity and ratings → scale [0,1]
        self.pop_norm = self._normalize(self.pop)
        self.rating_norm = self._normalize(self.rating)

        # Precompute embeddings for all docs to avoid recomputing on every query
        self.doc_embeddings = self.embedder.encode(self.texts, normalize_embeddings=True)

    def _normalize(self, arr):
        """Utility to normalize metadata values safely"""
        if len(arr) == 0:
            return np.zeros_like(arr)
        scaler = MinMaxScaler()
        return scaler.fit_transform(arr.reshape(-1, 1)).ravel()

    def recommend(self, user_pref_text, k=5, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Recommend items based on semantic similarity + popularity + rating.

        Args:
            user_pref_text (str): user preference text
            k (int): number of recommendations
            alpha (float): weight for semantic similarity
            beta (float): weight for popularity
            gamma (float): weight for ratings

        Returns:
            list of dicts with {'meta': ..., 'score': ...}
        """
        # Encode query (normalize to enable cosine similarity via dot product)
        q_emb = self.embedder.encode([user_pref_text], normalize_embeddings=True)
        q_emb = np.array(q_emb).squeeze()

        # Cosine similarity (dot product since embeddings are normalized)
        sims = self.doc_embeddings @ q_emb.T

        # Final weighted score
        final_score = alpha * sims + beta * self.pop_norm + gamma * self.rating_norm

        # Top-k results
        idxs = np.argsort(-final_score)[:k]
        results = [
            {"meta": self.docs[i]['meta'], "text": self.texts[i], "score": float(final_score[i])}
            for i in idxs
        ]
        return results
