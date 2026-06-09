import pickle
import pandas as pd
import numpy as np

class EmbeddingDataset:
    def __init__(self, path: str):
        with open(path, 'rb') as f:
            embeddings, targets = pickle.load(f)
        self._embeddings: np.ndarray = embeddings
        self._targets: pd.Series = targets
        assert len(self._embeddings) == len(self._targets)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self._embeddings, self._targets), f)

    def get_embedding(self) -> np.ndarray:
        return self._embeddings

    def get_target(self) -> pd.Series:
        return self._targets

    def __len__(self) -> int:
        return len(self._embeddings)