# reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    """
    Reranks candidate documents based on a CrossEncoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank a list of candidate documents based on relevance to the query.

        Args:
            query (str): user query
            candidates (list): list of dicts, each with 'meta' and 'text'

        Returns:
            List of dicts with added 'rerank_score', sorted by descending score
        """
        pairs = []

        for c in candidates:
            # Attempt to get text from 'text', 'item_name', or 'answer'
            text = c.get('text') or c.get('meta', {}).get('item_name') or c.get('meta', {}).get('answer') or ""
            pairs.append((query, text))

        scores = self.model.predict(pairs)

        for i, s in enumerate(scores):
            candidates[i]['rerank_score'] = float(s)

        # Sort descending by rerank score
        candidates_sorted = sorted(candidates, key=lambda x: -x['rerank_score'])
        return candidates_sorted
