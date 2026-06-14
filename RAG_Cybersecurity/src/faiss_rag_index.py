from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class Neighbor:
    index: int
    label: int
    similarity: float
    z_vector: np.ndarray

class FaissRAGIndex:

    def __init__(self) -> None:
        self.global_index = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []
        self.labels: Optional[np.ndarray] = None
        self.z_vectors: Optional[np.ndarray] = None
        self.cache_signature: Optional[str] = None

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS non è installato. Eseguire: pip install faiss-cpu") from exc
        return faiss

    @classmethod
    def _normalise_for_cosine(cls, vectors: np.ndarray) -> np.ndarray:
        faiss = cls._faiss()
        normalised = np.asarray(vectors, dtype=np.float32).copy()
        faiss.normalize_L2(normalised)
        return normalised

    def fit(self, X: pd.DataFrame, labels: np.ndarray, feature_names: list[str], cache_signature: Optional[str] = None) -> None:
        faiss = self._faiss()
        labels = np.asarray(labels, dtype=np.int64)
        if len(X) != len(labels):
            raise ValueError("Numero di righe ed etichette non coincidente.")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("L'indice richiede entrambe le classi 0 e 1.")

        X = X.loc[:, feature_names].copy()
        self.feature_names = list(feature_names)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        imputed = self.imputer.fit_transform(X)
        self.z_vectors = self.scaler.fit_transform(imputed).astype(np.float32)
        retrieval_vectors = self._normalise_for_cosine(self.z_vectors)

        dimension = retrieval_vectors.shape[1]
        self.global_index = faiss.IndexFlatIP(dimension)
        self.global_index.add(retrieval_vectors)
        self.labels = labels
        self.cache_signature = cache_signature

    def _check_ready(self) -> None:
        if self.global_index is None or self.imputer is None or self.scaler is None or self.labels is None or self.z_vectors is None:
            raise RuntimeError("Indice FAISS non costruito o non caricato.")

    def transform_query(self, row: pd.Series | pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        assert self.imputer is not None and self.scaler is not None

        if isinstance(row, pd.DataFrame):
            frame = row.loc[:, self.feature_names].copy()
        elif isinstance(row, pd.Series):
            frame = pd.DataFrame([row.reindex(self.feature_names).to_numpy(dtype=np.float64)], columns=self.feature_names)
        else:
            values = np.asarray(row, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[1] != len(self.feature_names):
                raise ValueError(f"La query ha {values.shape[1]} feature; attese {len(self.feature_names)}.")
            frame = pd.DataFrame(values, columns=self.feature_names)

        imputed = self.imputer.transform(frame)
        z_query = self.scaler.transform(imputed).astype(np.float32)
        retrieval_query = self._normalise_for_cosine(z_query)
        return z_query, retrieval_query

    def search(self, row: pd.Series | np.ndarray, k: int) -> list[Neighbor]:
        self._check_ready()
        assert self.global_index is not None and self.labels is not None and self.z_vectors is not None
        _, query = self.transform_query(row)
        real_k = min(max(1, int(k)), int(self.global_index.ntotal))
        similarities, indices = self.global_index.search(query, real_k)
        return [
            Neighbor(index=int(index), label=int(self.labels[index]), similarity=float(similarity), z_vector=self.z_vectors[index].copy())
            for similarity, index in zip(similarities[0], indices[0])
            if index >= 0
        ]

    def majority_vote(self, row: pd.Series | np.ndarray, k: int) -> tuple[int, dict[int, int], list[Neighbor]]:
        neighbors = self.search(row, k)
        if not neighbors:
            raise RuntimeError("FAISS non ha restituito vicini.")
        counts = {0: sum(n.label == 0 for n in neighbors), 1: sum(n.label == 1 for n in neighbors)}
        if counts[0] == counts[1]:
            prediction = int(neighbors[0].label)
        else:
            prediction = 0 if counts[0] > counts[1] else 1
        return prediction, counts, neighbors

    @staticmethod
    def _paths(prefix: str | os.PathLike[str]) -> dict[str, Path]:
        base = Path(prefix)
        return {"global": Path(str(base) + "_global.faiss"), "meta": Path(str(base) + "_metadata.pkl")}

    @classmethod
    def exists(cls, prefix: str | os.PathLike[str]) -> bool:
        return all(path.exists() for path in cls._paths(prefix).values())

    def save(self, prefix: str | os.PathLike[str]) -> None:
        faiss = self._faiss()
        self._check_ready()
        paths = self._paths(prefix)
        paths["meta"].parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.global_index, str(paths["global"]))
        with paths["meta"].open("wb") as handle:
            pickle.dump(
                {
                    "feature_names": self.feature_names,
                    "labels": self.labels,
                    "z_vectors": self.z_vectors,
                    "imputer": self.imputer,
                    "scaler": self.scaler,
                    "cache_signature": self.cache_signature,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, prefix: str | os.PathLike[str]) -> None:
        faiss = self._faiss()
        paths = self._paths(prefix)
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError(f"Cache FAISS incompleta: {prefix!s}")
        self.global_index = faiss.read_index(str(paths["global"]))
        with paths["meta"].open("rb") as handle:
            metadata = pickle.load(handle)
        self.feature_names = list(metadata["feature_names"])
        self.labels = np.asarray(metadata["labels"], dtype=np.int64)
        self.z_vectors = np.asarray(metadata["z_vectors"], dtype=np.float32)
        self.imputer = metadata["imputer"]
        self.scaler = metadata["scaler"]
        self.cache_signature = metadata.get("cache_signature")