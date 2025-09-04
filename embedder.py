# embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
import random
from googletrans import Translator
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Download required NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.translator = Translator()
        
        # Load T5 for paraphrasing
        self.t5_model_name = "t5-small"
        self.t5_tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
        self.t5_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.t5_model.to(self.t5_device)

    # -------------------- Preprocessing --------------------
    def preprocess(self, text: str) -> str:
        # Lowercase and remove non-alphanumeric characters
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Tokenize, remove stopwords, lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    # -------------------- Augmentation Methods --------------------
    def synonym_replacement(self, sentence: str, n: int = 2) -> str:
        words = sentence.split()
        if not words:
            return sentence
        new_words = words.copy()
        random_words = random.sample(words, min(n, len(words)))
        for word in random_words:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym_words = synonyms[0].lemma_names()
                if synonym_words and synonym_words[0].lower() != word.lower():
                    new_words[words.index(word)] = synonym_words[0].replace('_', ' ')
        return ' '.join(new_words)
    
    def back_translate(self, sentence: str, src='en', mid='fr') -> str:
        try:
            translated = self.translator.translate(sentence, src=src, dest=mid).text
            back_translated = self.translator.translate(translated, src=mid, dest=src).text
            return back_translated
        except Exception:
            return sentence  # fallback if translation fails
    
    def paraphrase_t5(self, sentence: str, max_length: int = 64) -> str:
        text = "paraphrase: " + sentence + " </s>"
        encoding = self.t5_tokenizer.encode_plus(text, return_tensors="pt").to(self.t5_device)
        outputs = self.t5_model.generate(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.5
        )
        paraphrased = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased

    # -------------------- Main Encode Method --------------------
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
        augment: bool = False,
        augment_methods: List[str] = None
    ) -> np.ndarray:
        # Ensure list
        if isinstance(texts, str):
            texts = [texts]
        
        if augment and augment_methods is None:
            augment_methods = ['synonym', 'back_translate', 't5']
        
        processed_texts = []
        for t in texts:
            t_proc = self.preprocess(t)
            
            if augment:
                for method in augment_methods:
                    if method == 'synonym':
                        t_proc = self.synonym_replacement(t_proc)
                    elif method == 'back_translate':
                        t_proc = self.back_translate(t_proc)
                    elif method == 't5':
                        t_proc = self.paraphrase_t5(t_proc)
            
            processed_texts.append(t_proc)
        
        # Encode
        emb = self.model.encode(processed_texts, batch_size=batch_size, show_progress_bar=True)
        emb = np.array(emb, dtype='float32')
        
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb = emb / norms
        
        return emb
