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
        """
        Carica il dataset serializzato con Dataset.save() o TextDataset.save().
        Il file deve contenere una tupla (list_of_texts, target_series).
        """
        with open(path, 'rb') as f:
            texts, targets = pickle.load(f)
        self._texts: List[str] = texts
        self._targets: pd.Series = targets
        assert len(self._texts) == len(self._targets), "Mismatch tra testi e target"

    @classmethod
    def from_data(cls, texts: List[str], targets: pd.Series) -> 'TextDataset':
        """
        Crea un'istanza di TextDataset direttamente da liste di testi e target.
        Utile per conversioni senza passare da file.
        """
        instance = cls.__new__(cls)
        instance._texts = texts
        instance._targets = targets
        return instance

    @property
    def texts(self) -> List[str]:
        """Restituisce la lista dei testi (sola lettura)."""
        return self._texts

    @property
    def targets(self) -> pd.Series:
        """Restituisce la Series delle etichette (sola lettura)."""
        return self._targets

    def save(self, path: str):
        """
        Serializza il dataset testuale in un file pickle.
        Il file conterrà una tupla (list_of_texts, target_series).
        """
        with open(path, 'wb') as f:
            pickle.dump((self._texts, self._targets), f)

    def text_to_emb(self, emb: Embedding) -> EmbeddingDataset:
        """
        Converte tutti i testi in embedding e restituisce un EmbeddingDataset.
        Args:
            emb: istanza di Embedding già caricata.
        Returns:
            EmbeddingDataset contenente le matrici di embedding e le etichette.
        """
        embeddings = emb.encode(self._texts)
        # Crea un'istanza di EmbeddingDataset senza passare da file
        emb_dataset = EmbeddingDataset.__new__(EmbeddingDataset)
        emb_dataset._embeddings = embeddings
        emb_dataset._targets = self._targets
        return emb_dataset