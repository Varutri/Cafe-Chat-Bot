RAG-Based Chatbot for a Café
Objective:
Develop a chatbot that answers questions about:
-Shop timings
-Menu items
-Meal costs
-Meal suggestions based on preferences
Requirements:
-Use a small document store (FAQs, menu, hours).
-Implement retrieval using FAISS or ChromaDB.
-Use a lightweight LLM (MiniLM, DistilBERT) for generation.
-Deploy via CLI or web interface (Gradio, Streamlit).
Skills Tested:
-Embedding and retrieval
-LLM integration
-Bot interface design


PROCESS FOLLOWED FOR CREATING EXTERNAL KNOWLEDGE BASE:- 





<img width="241" height="261" alt="image" src="https://github.com/user-attachments/assets/b0598344-6fc5-4bce-a312-a000402fca6c" />

















First Photo of my chat bot that also works as a RLHF (REINFORCEMENT LEARNING with HUMAN FEEDBACK)








<img width="509" height="422" alt="image" src="https://github.com/user-attachments/assets/f96043a3-8bdb-43e0-82c0-beb498b9586e" />




















Second Photo










<img width="518" height="405" alt="image" src="https://github.com/user-attachments/assets/2b794d8b-80c9-4a9f-89fa-ae06b1a9f06d" />





















After updating the prompt 











<img width="474" height="305" alt="image" src="https://github.com/user-attachments/assets/5f72975b-28cb-46e5-99b7-d3e9f8e6724c" />










































How I Built My Café Chatbot Using RAG (Retrieval-Augmented Generation)

It all started with a simple idea:
"Wouldn’t it be cool if a café could have a chatbot that not only answers FAQs like opening hours, but also recommends dishes, understands customer preferences, and gives natural, human-like responses?"

At first, it sounded simple. But the deeper I went, the more I realized — building a smart assistant is not just about plugging in an LLM. It’s about designing a pipeline where every step works together like gears in a machine.

🏗️ Step 1: Building the Knowledge Base

The café had three sources of truth:

Menu items 🍰

Order history 📊

FAQs ❓

I combined them into a document store where each entry had both text and metadata (like number of orders or average rating).

👉 Why?
Because customers don’t just want to know what a “Latte” is — they want to know if it’s popular, if people like it, and if it matches their preferences. So metadata became just as important as raw text.

🧠 Step 2: Choosing the Right Embeddings

For dense embeddings, I picked MiniLM (all-MiniLM-L6-v2).

👉 Why MiniLM?

Lightweight and super-fast ⚡

Captures semantic meaning well (great balance of performance vs accuracy)

Works beautifully with FAISS indexing

Every document and FAQ was preprocessed, lemmatized, and encoded into embeddings. To make it stronger, I even added data augmentation (synonym replacement, back translation, and T5-based paraphrasing).

This ensured that no matter how a customer phrased their query — “Tell me about cappuccino” vs “What is a cappuccino?” — the chatbot would still retrieve the right info.

🔍 Step 3: Indexing with FAISS

I needed a fast way to search through vectors, so I used FAISS with Inner Product (IP).

👉 Why Inner Product?
Because I normalized all embeddings, inner product ≈ cosine similarity.
This meant retrieval was quick and accurate, without being biased toward longer vectors.

Saving metadata alongside vectors ensured that when I retrieved results, I could still reference whether the doc was an FAQ, menu item, or something else.

🎯 Step 4: Retrieval – Going Hybrid

Here’s where I hit a challenge.
Dense embeddings are great at capturing semantic meaning, but sometimes customers ask keyword-heavy queries.

Example:
“Do you serve gluten-free muffins?”

Dense search might not always pick this up if “gluten-free” is rare.

So I introduced a Hybrid Retriever:

Dense search via FAISS

Sparse search via TF-IDF

Weighted combination of both

👉 Why Hybrid?
Because I didn’t want my bot to miss obvious keywords. This gave me the best of both worlds: understanding semantics and exact matches.

⭐ Step 5: Recommendations

I didn’t want the bot to just answer questions. I wanted it to recommend dishes like a real barista.

So I built a hybrid recommender that combined:

Semantic similarity (what you typed)

Popularity (number of orders)

Ratings (average feedback)

👉 Why?
Because if 90% of customers love our cappuccino, the bot should be smart enough to nudge you toward it — while still respecting your preferences like “I want something light and sweet.”

📌 Step 6: Reranking with CrossEncoder

Retrieval gave me candidates, but I wanted precision. That’s where I used a Cross-Encoder (MiniLM Cross-Encoder) to rerank results.

👉 Why Reranker?
Because dense retrieval is like “guessing which shelves to check in a library,” while reranking is like “reading those books to see which really answers the question.”

This step made the bot far more accurate and reliable.

💡 Step 7: Response Generation

For natural answers, I chose FLAN-T5-small.

👉 Why FLAN-T5?

Lightweight, runs fast even on CPU

Strong at instruction-following

Generates polite, human-like responses

I crafted prompts carefully to control tone:

Always give 2–5 sentences

Be helpful and friendly

Personalize with user preferences

This ensured responses didn’t sound robotic. Instead, they felt like a real café assistant.

🎨 Step 8: Building the Frontend

Finally, I wrapped everything in a Streamlit app.

A clean interface for customers to chat

Sidebar for adding preferences (vegetarian, spicy, budget)

Feedback logging to improve responses

🚀 The Result

The café chatbot could now:
✔️ Answer FAQs like timings, menu items, or ingredients
✔️ Recommend dishes based on popularity, rating, and your taste
✔️ Personalize responses with preferences
✔️ Provide natural, friendly explanations

All of this was powered by RAG (Retrieval-Augmented Generation), carefully designed to balance speed, accuracy, and usability.

✨ What I Learned

Embeddings matter → MiniLM gave me the sweet spot of speed + accuracy.

Hybrid retrieval is powerful → Never rely on just dense or sparse.

Rerankers bring precision → They clean up noisy retrieval.

Small LLMs can be mighty → FLAN-T5-small was more than enough for controlled generation.

Metadata is gold → Popularity & ratings gave the bot “human” intelligence.

💭 Looking back, this project was more than building a chatbot. It was about designing an ecosystem where every component — indexer, retriever, recommender, reranker, generator — plays a unique role.

That’s the beauty of RAG: it’s not just about using an LLM, it’s about teaching it where to look, how to rank, and how to talk.

🔥 And that’s how my café chatbot was born.
Now, whether a customer asks “What’s your best latte?” or “I want something refreshing under $5”… the bot knows exactly what to say.
