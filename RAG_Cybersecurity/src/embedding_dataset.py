import pickle
import pandas as pd
import numpy as np

class EmbeddingDataset:
    """
    Dataset di embedding: contiene gli embedding vettoriali e le etichette.
    Attributi privati:
        _embeddings (np.ndarray): matrice degli embedding, shape (n_samples, dim)
        _targets (pd.Series): serie delle etichette (0/1).
    """

    def __init__(self, path: str):
        with open(path, 'rb') as f:
            embeddings, targets = pickle.load(f)
        self._embeddings: np.ndarray = embeddings
        self._targets: pd.Series = targets
        assert len(self._embeddings) == len(self._targets), "Mismatch tra embeddings e target"

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self._embeddings, self._targets), f)

    def get_embedding(self) -> np.ndarray:
        """Restituisce la matrice degli embedding (sola lettura)."""
        return self._embeddings

    def get_target(self) -> pd.Series:
        """Restituisce la Series delle etichette (sola lettura)."""
        return self._targets

    def __len__(self) -> int:
        return len(self._embeddings)