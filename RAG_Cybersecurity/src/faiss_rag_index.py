from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.embedding import EmbeddingModel
from src.feature_text_encoder import FeatureTextEncoder


@dataclass(frozen=True)
class Neighbor:
    index: int
    label: int
    similarity: float


class FaissRAGIndex:
    def __init__(
        self,
        model_name: str,
        embedding_batch_size: int,
        sample_batch_size: int,
        top_features_per_sample: int,
        features_per_chunk: int,
        device: str | None,
        show_progress: bool,
    ) -> None:
        self.model_name = model_name
        self.embedding_batch_size = int(embedding_batch_size)
        self.sample_batch_size = int(sample_batch_size)
        self.top_features_per_sample = int(top_features_per_sample)
        self.features_per_chunk = int(features_per_chunk)
        self.device = device
        self.show_progress = bool(show_progress)

        self.index = None
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.labels: np.ndarray | None = None
        self.empty_feature_names: list[str] = []
        self.cache_signature: str | None = None

        self._model: EmbeddingModel | None = None
        self._encoder: FeatureTextEncoder | None = None

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Installare FAISS con: pip install faiss-cpu") from exc
        return faiss

    def _get_model(self) -> EmbeddingModel:
        if self._model is None:
            self._model = EmbeddingModel(self.model_name, self.device)
        return self._model

    def _get_encoder(self) -> FeatureTextEncoder:
        if self._encoder is None:
            self._encoder = FeatureTextEncoder(
                self.feature_names,
                self.top_features_per_sample,
                self.features_per_chunk,
            )
        return self._encoder

    def _encode_matrix(self, matrix: np.ndarray, description: str) -> np.ndarray:
        model = self._get_model()
        encoder = self._get_encoder()
        starts = range(0, len(matrix), self.sample_batch_size)
        iterator = tqdm(
            starts,
            desc=description,
            unit="batch",
            disable=not self.show_progress,
        )

        output: list[np.ndarray] = []
        for start in iterator:
            batch = matrix[start:start + self.sample_batch_size]
            chunks_by_row = [encoder.row_to_chunks(row) for row in batch]
            flat_chunks = [chunk for chunks in chunks_by_row for chunk in chunks]
            chunk_vectors = model.encode(flat_chunks, self.embedding_batch_size)

            offset = 0
            for chunks in chunks_by_row:
                current = chunk_vectors[offset:offset + len(chunks)]
                vector = current.mean(axis=0)
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise RuntimeError("Embedding medio non valido.")
                output.append((vector / norm).astype(np.float32))
                offset += len(chunks)

        result = np.vstack(output).astype(np.float32)
        if len(result) != len(matrix):
            raise RuntimeError("Numero di embedding non coerente con i campioni.")
        return result

    def fit(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        cache_signature: str,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(X) != len(labels) or X.empty:
            raise ValueError("Training set non valido.")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("Il training deve contenere entrambe le classi.")

        self.feature_names = X.columns.astype(str).tolist()
        self.labels = labels.copy()
        self.cache_signature = cache_signature
        self._encoder = None

        prepared = X.copy()
        self.empty_feature_names = [
            name for name in self.feature_names if prepared[name].isna().all()
        ]
        if self.empty_feature_names:
            prepared.loc[:, self.empty_feature_names] = 0.0

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        imputed = self.imputer.fit_transform(prepared)
        scaled = self.scaler.fit_transform(imputed).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise RuntimeError("Il training standardizzato non è valido.")

        vectors = self._encode_matrix(scaled, "Embedding training")
        faiss = self._faiss()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _prepare_queries(self, X: pd.DataFrame) -> np.ndarray:
        if self.index is None or self.imputer is None or self.scaler is None:
            raise RuntimeError("Indice non costruito o non caricato.")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Feature mancanti nel test: {missing[:10]}")

        ordered = X[self.feature_names].copy()
        if self.empty_feature_names:
            ordered.loc[:, self.empty_feature_names] = ordered.loc[
                :, self.empty_feature_names
            ].fillna(0.0)
        imputed = self.imputer.transform(ordered)
        scaled = self.scaler.transform(imputed).astype(np.float32)
        return self._encode_matrix(scaled, "Embedding test")

    def search_many(self, X: pd.DataFrame, k: int) -> list[list[Neighbor]]:
        if self.index is None or self.labels is None:
            raise RuntimeError("Indice non disponibile.")
        if k <= 0:
            raise ValueError("k deve essere maggiore di zero.")

        queries = self._prepare_queries(X)
        real_k = min(int(k), int(self.index.ntotal))
        similarities, indices = self.index.search(queries, real_k)

        return [
            [
                Neighbor(int(index), int(self.labels[index]), float(similarity))
                for similarity, index in zip(row_sim, row_idx)
                if int(index) >= 0
            ]
            for row_sim, row_idx in zip(similarities, indices)
        ]

    @staticmethod
    def hybrid_vote(
        neighbors: list[Neighbor],
        majority_weight: float,
    ) -> int:
        if not neighbors:
            raise RuntimeError("Nessun vicino restituito da FAISS.")
        if not 0.0 <= majority_weight <= 1.0:
            raise ValueError("HYBRID_MAJORITY_WEIGHT deve essere tra 0 e 1.")

        count_scores = {
            label: sum(n.label == label for n in neighbors) / len(neighbors)
            for label in (0, 1)
        }

        # Le similarità cosine negative non forniscono evidenza utile.
        # Le similarità positive vengono sommate per classe.
        weights = [max(0.0, float(n.similarity)) for n in neighbors]
        total_weight = sum(weights)
        if total_weight == 0.0:
            similarity_scores = count_scores.copy()
        else:
            similarity_scores = {
                label: sum(
                    weight for neighbor, weight in zip(neighbors, weights)
                    if neighbor.label == label
                ) / total_weight
                for label in (0, 1)
            }

        similarity_weight = 1.0 - majority_weight
        combined = {
            label: (
                majority_weight * count_scores[label]
                + similarity_weight * similarity_scores[label]
            )
            for label in (0, 1)
        }

        if combined[0] == combined[1]:
            return int(neighbors[0].label)
        return 0 if combined[0] > combined[1] else 1

    def predict_many(
        self,
        X: pd.DataFrame,
        k: int,
        majority_weight: float,
    ) -> np.ndarray:
        all_neighbors = self.search_many(X, k)
        return np.asarray(
            [self.hybrid_vote(neighbors, majority_weight) for neighbors in all_neighbors],
            dtype=np.int64,
        )

    @staticmethod
    def _paths(prefix: str | Path) -> tuple[Path, Path]:
        base = Path(prefix)
        return Path(str(base) + ".faiss"), Path(str(base) + "_metadata.pkl")

    @classmethod
    def exists(cls, prefix: str | Path) -> bool:
        faiss_path, pkl_path = cls._paths(prefix)
        return faiss_path.is_file() and pkl_path.is_file()

    def save(self, prefix: str | Path) -> None:
        if self.index is None or self.imputer is None or self.scaler is None:
            raise RuntimeError("Nessun indice da salvare.")
        faiss_path, pkl_path = self._paths(prefix)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)

        self._faiss().write_index(self.index, str(faiss_path))
        metadata = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "labels": self.labels,
            "empty_feature_names": self.empty_feature_names,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "cache_signature": self.cache_signature,
            "top_features_per_sample": self.top_features_per_sample,
            "features_per_chunk": self.features_per_chunk,
        }
        with pkl_path.open("wb") as file:
            pickle.dump(metadata, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, prefix: str | Path) -> None:
        faiss_path, pkl_path = self._paths(prefix)
        self.index = self._faiss().read_index(str(faiss_path))
        with pkl_path.open("rb") as file:
            metadata = pickle.load(file)

        self.model_name = str(metadata["model_name"])
        self.feature_names = list(metadata["feature_names"])
        self.labels = np.asarray(metadata["labels"], dtype=np.int64)
        self.empty_feature_names = list(metadata.get("empty_feature_names", []))
        self.imputer = metadata["imputer"]
        self.scaler = metadata["scaler"]
        self.cache_signature = metadata.get("cache_signature")
        self.top_features_per_sample = int(metadata["top_features_per_sample"])
        self.features_per_chunk = int(metadata["features_per_chunk"])
        self._model = None
        self._encoder = None
