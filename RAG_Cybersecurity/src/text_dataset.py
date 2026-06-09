import pickle
import pandas as pd
import numpy as np
from typing import List
from embedding import Embedding
from embedding_dataset import EmbeddingDataset

class TextDataset:
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            texts, targets = pickle.load(f)
        self._texts: List[str] = texts
        self._targets: pd.Series = targets
        assert len(self._texts) == len(self._targets)

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

    def text_to_emb(self, emb: Embedding, batch_size: int = 64) -> EmbeddingDataset:
        embeddings = emb.encode(self._texts, batch_size=batch_size)
        emb_dataset = EmbeddingDataset.__new__(EmbeddingDataset)
        emb_dataset._embeddings = embeddings
        emb_dataset._targets = self._targets
        return emb_dataset