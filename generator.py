# generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Dict, Optional, Union

class Generator:
    """
    Wrapper around a seq2seq model (e.g., FLAN-T5) for controlled response generation.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[Union[int, str]] = None):
        # Determine device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device != -1:
            self.model = self.model.to(device)

        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def craft_prompt(self, query: str, retrieved_docs: List[Dict], user_pref: Optional[str] = None) -> str:
        """
        Build a concise context for the generator.
        Keeps token budget small while combining FAQs, items, and text docs.
        """
        ctx = []

        for d in retrieved_docs[:4]:
            meta = d.get("meta", {})
            if meta.get("type") == "faq":
                ctx.append(f"FAQ: Q: {meta.get('question')} A: {meta.get('answer')}")
            elif meta.get("type") == "item":
                ctx.append(f"Item: {meta.get('item_name')} "
                           f"(orders: {meta.get('num_orders', 0)}, rating: {meta.get('avg_rating', 0.0)})")
            else:
                ctx.append(d.get("text", ""))

        context = "\n---\n".join(ctx) if ctx else "No relevant context available."

        pref_section = f"\nUser preferences: {user_pref}" if user_pref else ""

        prompt = (
            "You are a friendly cafe assistant. "
            "Your task:- Always give a helpful explanation, not just a one-word reply.- Use complete sentences (2–5 sentences).- Base your answer on the context provided below.- If preferences are given, personalize the answer to match them.- If you are unsure, say so politely and suggest what the customer could try instead.\n\n"
            f"Context:\n{context}{pref_section}\n\n"
            f"Question: {query}\nAnswer:"
        )
        return prompt

    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.2, num_beams: int = 3) -> str:
        """
        Generate a response given a prompt.
        """
        out = self.pipe(
            prompt,
            max_length=max_length,
            do_sample=False,  # deterministic output
            num_beams=num_beams
        )[0]["generated_text"]

        return out
