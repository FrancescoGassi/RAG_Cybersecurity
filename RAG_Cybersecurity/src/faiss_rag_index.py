from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.embedding import EmbeddingModel
from src.feature_text_encoder import FeatureTextEncoder
from src.config import RETRIEVAL_MODE


@dataclass(frozen=True)
class Neighbor:
    index: int
    label: int
    similarity: float
    vector: np.ndarray


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
        self.feature_names: list[str] = []
        self.labels: np.ndarray | None = None
        self.cache_signature: str | None = None
        self.training_vectors: np.ndarray | None = None

        self._model: EmbeddingModel | None = None
        self._encoder: FeatureTextEncoder | None = None

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Installare FAISS con: pip install faiss-cpu") from exc
        return faiss

    @staticmethod
    def _normalize_inplace(arr: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(arr)
        FaissRAGIndex._faiss().normalize_L2(arr)
        return arr

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

        # Converti a numpy e controlla presenza di NaN
        raw = X.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.any(np.isnan(raw)):
            raise ValueError("Il training contiene valori mancanti (NaN). Assicurarsi che i dati siano già preprocessati.")
        if not np.isfinite(raw).all():
            raise ValueError("Il training contiene valori infiniti.")
        self.training_vectors = raw.astype(np.float32)

        faiss = self._faiss()
        if RETRIEVAL_MODE == "original":
            print("   [ORIGINAL] Utilizzo vettori numerici originali (k-NN puro con similarità coseno)")
            vectors = np.ascontiguousarray(raw.astype(np.float32))
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        else:
            print("   [EMBEDDING] Utilizzo embedding (SentenceTransformer)")
            vectors = self._encode_matrix(raw, "Embedding training")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    def _validate_vector(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != len(self.feature_names):
            raise ValueError("Numero di feature diverso da quello del training.")
        if np.any(np.isnan(vector)):
            raise ValueError("Il vettore contiene valori mancanti (NaN).")
        if not np.isfinite(vector).all():
            raise ValueError("Il vettore contiene valori infiniti.")
        return vector.reshape(-1)

    def search_one(self, vector: np.ndarray, k: int) -> list[Neighbor]:
        if self.index is None or self.labels is None or self.training_vectors is None:
            raise RuntimeError("Indice non disponibile.")
        if k <= 0:
            raise ValueError("k deve essere maggiore di zero.")

        query_raw = self._validate_vector(vector)

        if RETRIEVAL_MODE == "original":
            query_vec = np.ascontiguousarray(query_raw.astype(np.float32)).reshape(1, -1)
            import faiss
            faiss.normalize_L2(query_vec)
        else:
            encoder = self._get_encoder()
            chunks = encoder.row_to_chunks(query_raw)
            chunk_vectors = self._get_model().encode(chunks, self.embedding_batch_size)
            query_embedding = chunk_vectors.mean(axis=0)
            norm = float(np.linalg.norm(query_embedding))
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError("Embedding query non valido.")
            query_vec = (query_embedding / norm).astype(np.float32).reshape(1, -1)

        real_k = min(int(k), int(self.index.ntotal))
        similarities, indices = self.index.search(query_vec, real_k)

        neighbors = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            neighbors.append(
                Neighbor(
                    index=int(idx),
                    label=int(self.labels[idx]),
                    similarity=float(sim),
                    vector=self.training_vectors[idx].copy(),
                )
            )
        return neighbors

    def search_many(self, X: pd.DataFrame, k: int) -> list[list[Neighbor]]:
        if self.index is None or self.labels is None or self.training_vectors is None:
            raise RuntimeError("Indice non disponibile.")
        if k <= 0:
            raise ValueError("k deve essere maggiore di zero.")

        queries = self._prepare_queries(X)
        real_k = min(int(k), int(self.index.ntotal))
        similarities, indices = self.index.search(queries, real_k)

        result = []
        for row_sim, row_idx in zip(similarities, indices):
            neighbors = []
            for sim, idx in zip(row_sim, row_idx):
                if idx < 0:
                    continue
                neighbors.append(
                    Neighbor(
                        index=int(idx),
                        label=int(self.labels[idx]),
                        similarity=float(sim),
                        vector=self.training_vectors[idx].copy(),
                    )
                )
            result.append(neighbors)
        return result

    def _prepare_queries(self, X: pd.DataFrame) -> np.ndarray:
        if self.index is None:
            raise RuntimeError("Indice non costruito.")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Feature mancanti nel test: {missing[:10]}")

        ordered = X[self.feature_names].to_numpy(dtype=np.float64)
        if np.any(np.isnan(ordered)):
            raise ValueError("Il test contiene valori mancanti (NaN).")
        if not np.isfinite(ordered).all():
            raise ValueError("Il test contiene valori infiniti.")
        ordered = ordered.astype(np.float32)

        if RETRIEVAL_MODE == "original":
            ordered = np.ascontiguousarray(ordered)
            import faiss
            faiss.normalize_L2(ordered)
            return ordered
        else:
            return self._encode_matrix(ordered, "Embedding test")

    # Voto di maggioranza puro (solo conteggio)
    def majority_vote(self, vector: np.ndarray, k: int) -> tuple[int, dict[int, int], list[Neighbor]]:
        neighbors = self.search_one(vector, k)
        if not neighbors:
            raise RuntimeError("Nessun vicino trovato.")
        counts = {0: 0, 1: 0}
        for n in neighbors:
            counts[n.label] += 1
        if counts[0] == counts[1]:
            pred = neighbors[0].label
        else:
            pred = 0 if counts[0] > counts[1] else 1
        return pred, counts, neighbors

    def predict_many(self, X: pd.DataFrame, k: int) -> np.ndarray:
        all_neighbors = self.search_many(X, k)
        predictions = []
        for neighbors in all_neighbors:
            if not neighbors:
                raise RuntimeError("Nessun vicino trovato per una riga.")
            counts = {0: 0, 1: 0}
            for n in neighbors:
                counts[n.label] += 1
            if counts[0] == counts[1]:
                pred = neighbors[0].label
            else:
                pred = 0 if counts[0] > counts[1] else 1
            predictions.append(pred)
        return np.asarray(predictions, dtype=np.int64)

    @staticmethod
    def _paths(prefix: str | Path) -> tuple[Path, Path]:
        base = Path(prefix)
        return Path(str(base) + ".faiss"), Path(str(base) + "_metadata.pkl")

    @classmethod
    def exists(cls, prefix: str | Path) -> bool:
        faiss_path, pkl_path = cls._paths(prefix)
        return faiss_path.is_file() and pkl_path.is_file()

    def save(self, prefix: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Nessun indice da salvare.")
        faiss_path, pkl_path = self._paths(prefix)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)

        self._faiss().write_index(self.index, str(faiss_path))
        metadata = {
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "labels": self.labels,
            "cache_signature": self.cache_signature,
            "top_features_per_sample": self.top_features_per_sample,
            "features_per_chunk": self.features_per_chunk,
            "training_vectors": self.training_vectors,
            "retrieval_mode": RETRIEVAL_MODE,
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
        self.cache_signature = metadata.get("cache_signature")
        self.top_features_per_sample = int(metadata["top_features_per_sample"])
        self.features_per_chunk = int(metadata["features_per_chunk"])
        self.training_vectors = metadata.get("training_vectors")
        
        saved_mode = metadata.get("retrieval_mode")
        if saved_mode and saved_mode != RETRIEVAL_MODE:
            print(f"   AVVISO: l'indice è stato salvato con modalità '{saved_mode}', ma ora è impostato '{RETRIEVAL_MODE}'. Ricostruire l'indice per coerenza.")
        
        self._model = None
        self._encoder = None