# cli.py
from data_loader import load_items, load_faq, load_orders, build_document_store
from embedder import Embedder
from indexer import FaissIndexer
from retriever import HybridRetriever
from reranker import Reranker
from generator import Generator
from recommender import SimpleRecommender
import json

def main():
    # Load data
    items = load_items(r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\Item_to_id.csv")
    faq = load_faq(r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\conversationo.csv")
    orders = load_orders(r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\food.csv")
    docs = build_document_store(items, faq, orders)

    # Initialize embedder
    embedder = Embedder()
    texts = [d['text'] for d in docs]
    vectors = embedder.encode(texts, normalize=True)

    # Build and save FAISS index
    indexer = FaissIndexer(dim=vectors.shape[1])
    indexer.build(vectors, [d['meta'] for d in docs])
    indexer.save()

    # Initialize retriever, reranker, generator, recommender
    retriever = HybridRetriever(docs, embedder, indexer)
    reranker = Reranker()
    generator = Generator()
    recommender = SimpleRecommender(docs, embedder)

    print("CafeBot ready. Type 'quit' to exit. Use 'recommend: <prefs>' for suggestions.")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break

        if q.lower().startswith("recommend:"):
            pref = q.split(":", 1)[1].strip()
            recs = recommender.recommend(pref, k=5)
            print("Recommended items:")
            for r in recs:
                print(r['meta'].get('item_name', "Unknown"), "score:", round(r['score'], 3))
            continue

        # Normal QA flow
        candidates = retriever.hybrid_search(q, k=8)

        # Attach texts to candidates
        for c in candidates:
            idx = c.get('index')
            if idx is not None and 0 <= idx < len(docs):
                c['text'] = docs[idx]['text']
            else:
                c['text'] = ""

        if not candidates:
            print("Bot: Sorry, I couldn't find relevant information.")
            continue

        # Rerank
        candidates = reranker.rerank(q, candidates)

        # Generate answer using top 4 candidates
        prompt = generator.craft_prompt(q, candidates[:4])
        answer = generator.generate(prompt)
        print("Bot:", answer)

        # Show top sources
        print("--- Sources ---")
        for c in candidates[:3]:
            print(json.dumps(c['meta'], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
