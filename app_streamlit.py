# src/app_streamlit.py
import streamlit as st
from data_loader import load_items, load_faq, load_orders, build_document_store
from embedder import Embedder
from indexer import FaissIndexer
from retriever import HybridRetriever
from reranker import reranker
from generator import Generator
from recommender import SimpleRecommender


@st.cache_resource
def prepare():
    items = load_items("data/items.csv")
    faq = load_faq("data/faq.csv")
    orders = load_orders("data/orders.csv")

    docs = build_document_store(items, faq, orders)

    embedder = Embedder()
    texts = [d["text"] for d in docs]
    vectors = embedder.encode(texts)

    indexer = FaissIndexer(dim=vectors.shape[1])
    indexer.build(vectors, [d["meta"] for d in docs])

    retriever = HybridRetriever(docs, embedder, indexer)
    reranker = reranker()
    generator = Generator()
    recommender = SimpleRecommender(docs, embedder)

    return docs, retriever, reranker, generator, recommender


docs, retriever, reranker, generator, recommender = prepare()

# --- UI ---
st.title("☕ Café Assistant")
st.write("Ask about shop timings, menu, prices, or get meal suggestions.")

st.caption(f"📚 Loaded {len(docs)} documents")

with st.sidebar:
    user_pref = st.text_input("Your preferences (vegetarian, spicy, budget, etc.)")
    use_pref_for_gen = st.checkbox("Attach preferences to answer", value=True)

q = st.text_input("Your question")
ask = st.button("Ask")

# --- Question flow ---
if ask and q.strip():
    st.subheader("🔍 Processing your question...")

    candidates = retriever.hybrid_search(q, k=10)
    for c in candidates:
        c["text"] = docs[c["index"]]["text"]

    reranked = reranker.rerank(q, candidates[:8])
    context_docs = reranked[:4]

    prompt = generator.craft_prompt(
        q, context_docs, user_pref if use_pref_for_gen else None
    )
    answer = generator.generate(prompt)

    st.subheader("✅ Answer")
    st.write(answer)

    st.subheader("📖 Top sources")
    for r in reranked[:5]:
        st.write(r["meta"])

else:
    st.info("👋 Ask me about the café! (timings, menu, prices, or meal ideas)")

# --- Recommendations always shown ---
st.subheader("🍴 Quick recommendations")
recs = recommender.recommend(user_pref if user_pref else "popular", k=3)
for r in recs:
    st.write(r["meta"].get("item_name"), "| score:", round(r["score"], 3))
