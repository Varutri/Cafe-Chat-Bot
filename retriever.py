# src/retriever.py
import numpy as np
from sentence_transformers import util

class MyRetriever:
    def __init__(self, faiss_indexer):
        # correct: 4 spaces inside __init__
        self.indexer = faiss_indexer


class HybridRetriever:
    def __init__(self, docs, embedder: 'Embedder', faiss_indexer: 'FaissIndexer'):
        # docs: list of dicts with 'text' and 'meta'
        self.docs = docs
        self.texts = [d['text'] for d in docs]
        self.embedder = embedder
        self.indexer = faiss_indexer
        # TF-IDF for sparse recall
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000).fit(self.texts)
        self.tfidf_matrix = self.tfidf.transform(self.texts)

    def dense_search(self, query, k=10):
        q_emb = self.embedder.encode([query])  # normalized
        return self.indexer.search(q_emb, top_k=k)

    def sparse_search(self, query, k=10):
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().squeeze()
        top_idx = np.argsort(-scores)[:k]
        results = [{"score": float(scores[i]), "meta": self.docs[i]['meta'], "index": i} for i in top_idx if scores[i]>0]
        return results

    def hybrid_search(self, query, k=8, alpha=0.6):
        dense = self.dense_search(query, k=k*2)
        sparse = self.sparse_search(query, k=k*2)
        # combine by metadata index; collect scores
        combined = {}
        for r in dense:
            idx = r['index']
            combined.setdefault(idx, {"meta": r['meta'], "dense_score": r['score'], "sparse_score":0.0})
        for r in sparse:
            idx = r['index']
            combined.setdefault(idx, {"meta": r['meta'], "dense_score":0.0, "sparse_score": r['score']})
        results = []
        for idx, v in combined.items():
            score = alpha * v['dense_score'] + (1-alpha) * v['sparse_score']
            results.append({"index": idx, "meta": v['meta'], "score": score})
        results = sorted(results, key=lambda x: -x['score'])[:k]
        return results
