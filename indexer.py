# src/indexer.py
import faiss
import numpy as np
import pickle
import os

class FaissIndexer:
    def __init__(self, dim, index_path="models/faiss.index", meta_path="models/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
        self.metadata = []  # parallel list of metadata dicts

    def build(self, vectors, docs_meta):
        assert vectors.shape[0] == len(docs_meta)
        self.index.add(vectors)
        self.metadata = docs_meta

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, q_vector, top_k=5):
        # q_vector: (1, dim)
        D, I = self.index.search(q_vector, top_k)
        # return list of (score, metadata)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), "meta": self.metadata[idx], "index": int(idx)})
        return results
