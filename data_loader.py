# src/data_loader.py
import pandas as pd

def load_items(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\Item_to_id.csv"):
    df = pd.read_csv(path)
    df['item'] = df['item'].astype(str)
    return df

def load_faq(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\conversationo.csv"):
    df = pd.read_csv(path)
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    df['doc'] = df['question'] + " ||| " + df['answer']
    return df

def load_orders(path="C:\\Users\\KIIT\\OneDrive\\Desktop\\Chat Bot\\Dataset\\Chat Bot Dataset\\food.csv"):
    df = pd.read_csv(path)
    return df

def build_document_store(items_df, faq_df, orders_df):
    docs = []
    # menu items
    for _, r in items_df.iterrows():
        item_id = str(r['id'])
        text = r['item']
        meta = {"type": "item", "item_id": item_id, "item_name": r['item']}
        # join with orders if exists
        order_row = orders_df[orders_df['id'].astype(str) == item_id]
        if not order_row.empty:
            meta['num_orders'] = int(order_row['num_orders'].values[0])
            meta['avg_rating'] = float(order_row['avg_rating'].values[0])
        else:
            meta['num_orders'] = 0
            meta['avg_rating'] = 0.0
        docs.append({"id": f"item_{item_id}", "text": text, "meta": meta})

    # faqs
    for _, r in faq_df.iterrows():
        qid = f"faq_{_}"
        text = r['question'] + " " + r['answer']
        meta = {"type": "faq", "question": r['question'], "answer": r['answer']}
        docs.append({"id": qid, "text": text, "meta": meta})

    return docs
