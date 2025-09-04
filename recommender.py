# src/recommender.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util

from sklearn.preprocessing import MinMaxScaler

class SimpleRecommender:
    def __init__(self, docs, embedder: 'Embedder'):
        # docs are same as used in indexer
        self.docs = docs
        self.texts = [d['text'] for d in docs]
        self.embedder = embedder
        # extract popularity and rating arrays
        self.pop = np.array([d['meta'].get('num_orders',0) for d in docs], dtype=float)
        self.rating = np.array([d['meta'].get('avg_rating',0.0) for d in docs], dtype=float)
        # normalize
        self.scaler_pop = MinMaxScaler()
        self.scaler_rating = MinMaxScaler()
        if len(self.pop) > 0:
            self.pop_norm = self.scaler_pop.fit_transform(self.pop.reshape(-1,1)).squeeze()
        else:
            self.pop_norm = np.zeros_like(self.pop)
        if len(self.rating) > 0:
            self.rating_norm = self.scaler_rating.fit_transform(self.rating.reshape(-1,1)).squeeze()
        else:
            self.rating_norm = np.zeros_like(self.rating)

    def recommend(self, user_pref_text, k=5, alpha=0.5, beta=0.3, gamma=0.2):
        """
        alpha: semantic score weight, beta: popularity, gamma: rating
        """
        q_emb = self.embedder.encode([user_pref_text])
        # compute cosine similarity between q_emb and all item embeddings
        # we assume indexer or precomputed embeddings accessible; for simplicity, compute on the fly:
        all_emb = self.embedder.encode(self.texts)
        # inner products (embeddings are normalized)
        sims = (all_emb @ q_emb.T).squeeze()
        final_score = alpha * sims + beta * self.pop_norm + gamma * self.rating_norm
        idxs = np.argsort(-final_score)[:k]
        results = []
        for i in idxs:
            meta = self.docs[i]['meta']
            results.append({"meta": meta, "score": float(final_score[i])})
        return results
