# embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        # Ensure input is a list
        if isinstance(texts, str):
            texts = [texts]

        emb = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        emb = np.array(emb, dtype='float32')

        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb = emb / norms

        return emb
