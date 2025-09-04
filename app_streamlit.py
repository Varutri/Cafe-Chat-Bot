# app_streamlit.py
import streamlit as st
import json
from data_loader import load_items, load_faq, load_orders, build_document_store
from embedder import  Embedder
from indexer import FaissIndexer
from retriever import HybridRetriever
from reranker import  Reranker
from generator import Generator
from recommender import SimpleRecommender
import re

# --- Normalization function ---
def normalize(text):
    text = text.lower()
    text = text.replace(' u ', ' you ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Dataset paths
ITEMS_PATH = r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\Item_to_id.csv"
FAQ_PATH = r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\conversationo.csv"
ORDERS_PATH = r"C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\food.csv"

@st.cache_resource
def prepare():
    items = load_items(ITEMS_PATH)
    faq = load_faq(FAQ_PATH)
    orders = load_orders(ORDERS_PATH)
    docs = build_document_store(items, faq, orders)

    embedder = Embedder()
    texts = [d["text"] for d in docs]
    vectors = embedder.encode(texts, normalize=True)

    indexer = FaissIndexer(dim=vectors.shape[1])
    indexer.build(vectors, [d["meta"] for d in docs])

    retriever = HybridRetriever(docs, embedder, indexer)
    ranker = Reranker()
    generator = Generator()
    recommender = SimpleRecommender(docs, embedder)

    return docs, retriever, ranker, generator, recommender

# Load resources
docs, retriever, reranker, generator, recommender = prepare()

# --- UI ---
st.title("☕ Café Assistant")
st.write("Ask about shop timings, menu, prices, or get meal suggestions.")
st.caption(f"📚 Loaded {len(docs)} documents")

with st.sidebar:
    user_pref = st.text_input("Your preferences (vegetarian, spicy, budget, etc.)")
    use_pref_for_gen = st.checkbox("Attach preferences to answer", value=True)

# --- User question ---
q = st.text_input("Your question")
ask = st.button("Ask")

if ask and q.strip():
    q_norm = normalize(q)
    st.subheader("🔍 Processing your question...")

    # Retrieve candidates
    candidates = retriever.hybrid_search(q_norm, k=12)  # fetch more to include detailed answers
    for c in candidates:
        idx = c.get("index")
        if idx is not None and 0 <= idx < len(docs):
            c["text"] = docs[idx]["text"]
        else:
            c["text"] = ""

    # Rerank candidates
    reranked = reranker.rerank(q_norm, candidates[:10])
    context_docs = reranked[:6]  # give generator more context

    # Generate answer
    prompt = generator.craft_prompt(
        q, context_docs, user_pref if use_pref_for_gen else None
    )
    answer = generator.generate(prompt)

    # Display answer
    st.subheader("✅ Answer")
    st.write(answer)

    # --- Human feedback ---
    feedback = st.radio("Did this answer help you?", ("👍 Yes", "👎 No"))
    if st.button("Submit Feedback"):
        feedback_data = {
            "query": q,
            "answer": answer,
            "feedback": 1 if feedback == "👍 Yes" else 0,
            "context_docs": [c["meta"] for c in context_docs],
            "user_pref": user_pref
        }
        with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        st.success("Thanks for your feedback! 🌟")

    # Show top sources
    st.subheader("📖 Top sources")
    for r in reranked[:5]:
        st.write(r["meta"])

else:
    st.info("👋 Ask me about the café! (timings, menu, prices, or meal ideas)")

# --- Recommendations ---
st.subheader("🍴 Quick recommendations")
recs = recommender.recommend(user_pref if user_pref else "popular", k=3)
for r in recs:
    st.write(r["meta"].get("item_name", "Unknown"), "| score:", round(r["score"], 3))
