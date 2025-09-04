# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=64, normalize=True):
        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        emb = np.array(emb, dtype='float32')
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms==0]=1
            emb = emb / norms
        return emb
