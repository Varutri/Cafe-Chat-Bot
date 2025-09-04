# src/generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


class ggenerator:
    """
    Wrapper around a seq2seq model (e.g., FLAN-T5) for controlled response generation.
    """

    def __init__(self, model_name="google/flan-t5-small", device=None):
        # device selection
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

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

    def craft_prompt(self, query, retrieved_docs, user_pref=None):
        """
        Build a concise context for the generator.
        Keeps token budget small while combining FAQs, items, and text docs.

        Args:
            query (str): user question
            retrieved_docs (list[dict]): docs with 'meta' and 'text'
            user_pref (str, optional): user preferences
        """
        ctx = []
        for d in retrieved_docs[:4]:
            meta = d.get("meta", {})

            if meta.get("type") == "faq":
                ctx.append(
                    f"FAQ: Q: {meta.get('question')} A: {meta.get('answer')}"
                )
            elif meta.get("type") == "item":
                ctx.append(
                    f"Item: {meta.get('item_name')} "
                    f"(orders: {meta.get('num_orders')}, rating: {meta.get('avg_rating')})"
                )
            else:
                ctx.append(d.get("text", ""))

        context = "\n---\n".join(ctx)

        pref_section = f"\nUser preferences: {user_pref}" if user_pref else ""

        prompt = (
            "You are a friendly cafe assistant. "
            "Use the context to answer the user's question concisely.\n\n"
            f"Context:\n{context}{pref_section}\n\n"
            f"Question: {query}\nAnswer:"
        )
        return prompt

    def generate(self, prompt, max_length=150, temperature=0.2, num_beams=3):
        """
        Generate a response given a prompt.

        Args:
            prompt (str): input text prompt
            max_length (int): max tokens
            temperature (float): randomness (not used if do_sample=False)
            num_beams (int): beam search width
        """
        out = self.pipe(
            prompt,
            max_length=max_length,
            do_sample=False,
            num_beams=num_beams
        )[0]["generated_text"]

        return out
