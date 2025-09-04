# indexer.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any

class FaissIndexer:
    def __init__(self, dim: int, index_path: str = "models/faiss.index", meta_path: str = "models/meta.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
        self.metadata: List[Dict[str, Any]] = []  # parallel list of metadata dicts

    def build(self, vectors: np.ndarray, docs_meta: List[Dict[str, Any]]):
        assert vectors.shape[0] == len(docs_meta), "Vectors and metadata length mismatch"
        self.index.add(vectors)
        self.metadata = docs_meta

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError("FAISS index or metadata file not found.")

    def search(self, q_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        q_vector: np.ndarray of shape (1, dim)
        Returns a list of dicts: [{"score": float, "meta": dict, "index": int}, ...]
        """
        D, I = self.index.search(q_vector, top_k)
        results = []

        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append({"score": float(score), "meta": self.metadata[idx], "index": int(idx)})

        return results
