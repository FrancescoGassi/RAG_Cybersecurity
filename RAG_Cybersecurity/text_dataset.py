import pickle
import pandas as pd
import numpy as np
from typing import List
from embedding import Embedding
from embedding_dataset import EmbeddingDataset

class TextDataset:
    """
    Dataset testuale: contiene i testi (uno per campione) e le etichette.
    Attributi privati:
        _texts (List[str]): lista di stringhe, ciascuna rappresenta un campione.
        _targets (pd.Series): serie delle etichette (0/1).
    """

    def __init__(self, path: str):
        with open(path, 'rb') as f:
            texts, targets = pickle.load(f)
        self._texts: List[str] = texts
        self._targets: pd.Series = targets
        assert len(self._texts) == len(self._targets), "Mismatch tra testi e target"

    @classmethod
    def from_data(cls, texts: List[str], targets: pd.Series) -> 'TextDataset':
        instance = cls.__new__(cls)
        instance._texts = texts
        instance._targets = targets
        return instance

    def get_texts(self) -> List[str]:
        return self._texts

    def get_targets(self) -> pd.Series:
        return self._targets

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self._texts, self._targets), f)

    def text_to_emb(self, emb: Embedding) -> EmbeddingDataset:
        embeddings = emb.encode(self._texts)
        emb_dataset = EmbeddingDataset.__new__(EmbeddingDataset)
        emb_dataset._embeddings = embeddings
        emb_dataset._targets = self._targets
        return emb_dataset