from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedding:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model = self.load_model(model_name)

    @classmethod
    def load_model(cls, model_name: str = 'all-MiniLM-L6-v2'):
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        return SentenceTransformer(model_name, device="cpu")

    def encode(self, sentences: List[str]) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        return self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)