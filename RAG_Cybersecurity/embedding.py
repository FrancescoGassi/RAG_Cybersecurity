from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedding:
    """
    Gestisce il caricamento del modello di embedding e la conversione di testi in vettori.
    Il modello è condiviso come attributo di classe per non ricaricarlo ogni volta.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model = self.load_model(model_name)

    @classmethod
    def load_model(cls, model_name: str = 'all-MiniLM-L6-v2'):
        return SentenceTransformer(model_name)

    def encode(self, sentences: List[str]) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        return self._model.encode(sentences, convert_to_numpy=True)