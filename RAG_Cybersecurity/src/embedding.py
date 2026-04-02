from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class Embedding:
    """
    Gestisce il caricamento del modello di embedding e la conversione di testi in vettori.
    Il modello è condiviso come attributo di classe per non ricaricarlo ogni volta.
    """
    _model = None  # attributo di classe condiviso

    @classmethod
    def load_model(cls, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Carica il modello di sentence-transformers se non già caricato.
        Il modello di default è 'all-MiniLM-L6-v2' (piccolo, veloce, 384 dimensioni).
        """
        if cls._model is None:
            cls._model = SentenceTransformer(model_name)
        return cls._model

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Crea un'istanza di Embedding. Il modello viene caricato (se non ancora) alla prima istanza.
        """
        self.model_name = model_name
        self._model = self.load_model(model_name)

    def encode(self, sentences: Union[str, List[str]]) -> np.ndarray:
        """
        Converte una frase o una lista di frasi in embedding.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        return self._model.encode(sentences, convert_to_numpy=True)