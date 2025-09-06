import pandas as pd
import re
from typing import List, Dict

# --- Normalization function ---
def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(' u ', ' you ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Text chunking utility ---
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of words with optional overlap.
    Example: chunk_size=200, overlap=50 → each chunk has 200 words, 
             with 50 words carried over to the next chunk.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def load_items(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\Item_to_id.csv"):
    df = pd.read_csv(path)
    df['item'] = df['name'].astype(str).apply(normalize)
    return df


def load_faq(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\conversationo.csv"):
    df = pd.read_csv(path)
    df['question'] = df['Question'].astype(str).apply(normalize)
    df['answer'] = df['answer'].astype(str)

    # Deduplicate FAQs: keep the longest answer for duplicate questions
    df['answer_len'] = df['answer'].apply(len)
    df = df.sort_values('answer_len', ascending=False)
    df = df.drop_duplicates(subset=['question'], keep='first')
    df.drop(columns=['answer_len'], inplace=True)

    df['doc'] = df['question'] + " ||| " + df['answer']
    return df


def load_orders(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\food.csv"):
    df = pd.read_csv(path)
    return df


def build_document_store(items_df, faq_df, orders_df, chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
    docs = []

    # --- Menu items ---
    for _, r in items_df.iterrows():
        item_id = str(r['id'])
        text = r['item']
        meta = {"type": "item", "item_id": item_id, "item_name": r['item']}

        # Join with orders
        order_row = orders_df[orders_df['id'].astype(str) == item_id]
        if not order_row.empty:
            meta['num_orders'] = int(order_row['times_appeared'].values[0])
            meta['avg_rating'] = float(order_row['food_rating'].values[0])
        else:
            meta['num_orders'] = 0
            meta['avg_rating'] = 0.0

        docs.append({"id": f"item_{item_id}", "text": text, "meta": meta})

    # --- FAQs (now chunked) ---
    for idx, r in faq_df.iterrows():
        qid = f"faq_{idx}"
        full_text = r['question'] + " " + r['answer']
        meta = {"type": "faq", "question": r['question'], "answer": r['answer']}

        chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            docs.append({"id": f"{qid}_{i}", "text": chunk, "meta": meta})

    return docs
