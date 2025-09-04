# src/cli.py
from data_loader import load_items, load_faq, load_orders, build_document_store
from embedder import Embedder
from indexer import FaissIndexer
from retriever import HybridRetriever
from reranker import Reranker
from generator import Generator
from recommender import SimpleRecommender
import json

def main():
    items = load_items("data/items.csv")
    faq = load_faq("data/faq.csv")
    orders = load_orders("data/orders.csv")
    docs = build_document_store(items, faq, orders)

    embedder = Embedder()
    texts = [d['text'] for d in docs]
    vectors = embedder.encode(texts)

    indexer = FaissIndexer(dim=vectors.shape[1])
    indexer.build(vectors, [d['meta'] for d in docs])
    indexer.save()

    retriever = HybridRetriever(docs, embedder, indexer)
    reranker = Reranker()
    generator = Generator()
    recommender = SimpleRecommender(docs, embedder)

    print("CafeBot ready. Type 'quit' to exit. Use 'recommend: <prefs>' for suggestions.")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("quit","exit"):
            break
        if q.lower().startswith("recommend:"):
            pref = q.split(":",1)[1].strip()
            recs = recommender.recommend(pref, k=5)
            print("Recommended:")
            for r in recs:
                print(r['meta'].get('item_name'), "score:", r['score'])
            continue

        # normal QA flow
        candidates = retriever.hybrid_search(q, k=8)
        # attach texts
        for c in candidates:
            c['text'] = docs[c['index']]['text']
        candidates = reranker.rerank(q, candidates)
        prompt = generator.craft_prompt(q, candidates[:4])
        answer = generator.generate(prompt)
        print("Bot:", answer)
        print("--- Sources ---")
        for c in candidates[:3]:
            print(json.dumps(c['meta'], ensure_ascii=False))
