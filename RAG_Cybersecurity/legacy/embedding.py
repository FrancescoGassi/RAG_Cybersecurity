from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from typing import List
import logging

class Embedding:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model = self.load_model(model_name)

    @classmethod
    def load_model(cls, model_name: str = 'all-MiniLM-L6-v2'):
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name, device=device)

    def encode(self, sentences: List[str], batch_size: int = 64) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embs = self._model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embs)
        return np.vstack(embeddings)