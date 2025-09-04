# src/reranker.py
from sentence_transformers import CrossEncoder

class reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, candidates):
        # candidates: list of dicts with 'meta' and the original doc text accessible in meta or by index
        pairs = []
        for c in candidates:
            # we need the doc text; expect candidates to carry 'text' too. If not, adapt.
            text = c.get('text') or c['meta'].get('item_name') or c['meta'].get('answer') or ""
            pairs.append((query, text))
        scores = self.model.predict(pairs)
        for i, s in enumerate(scores):
            candidates[i]['rerank_score'] = float(s)
        candidates = sorted(candidates, key=lambda x: -x['rerank_score'])
        return candidates
