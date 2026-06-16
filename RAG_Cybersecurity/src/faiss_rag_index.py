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
from tqdm import tqdm

from legacy.embedding import Embedding
from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_ENFORCE_MAX_LENGTH,
    EMBEDDING_FEATURES_PER_CHUNK,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SAMPLE_BATCH_SIZE,
    EMBEDDING_SHOW_PROGRESS,
    EMBEDDING_SORT_FEATURES_BY_MAGNITUDE,
    EMBEDDING_TOP_FEATURES_PER_SAMPLE,
    EMBEDDING_VALUE_DECIMALS,
)
from src.feature_names import BODMASFeatureNames
from src.feature_text_encoder import FeatureTextEncoder


@dataclass(frozen=True)
class Neighbor:
    index: int
    label: int
    similarity: float
    z_vector: np.ndarray


@dataclass(frozen=True)
class VoteResult:
    prediction: int
    counts: dict[int, int]
    similarity_scores: dict[int, float]
    neighbors: list[Neighbor]
    strategy: str


class FaissRAGIndex:
    def __init__(
        self,
        embedding_model_name: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_sample_batch_size: int | None = None,
        top_features_per_sample: int | None = None,
        features_per_chunk: int | None = None,
        value_decimals: int | None = None,
        sort_features_by_magnitude: bool | None = None,
        enforce_max_length: bool | None = None,
        device: str | None = None,
        show_progress_bar: bool | None = None,
    ) -> None:
        self.embedding_model_name = str(
            embedding_model_name
            if embedding_model_name is not None
            else EMBEDDING_MODEL_NAME
        ).strip()
        self.embedding_batch_size = int(
            embedding_batch_size
            if embedding_batch_size is not None
            else EMBEDDING_BATCH_SIZE
        )
        self.embedding_sample_batch_size = int(
            embedding_sample_batch_size
            if embedding_sample_batch_size is not None
            else EMBEDDING_SAMPLE_BATCH_SIZE
        )
        self.top_features_per_sample = int(
            top_features_per_sample
            if top_features_per_sample is not None
            else EMBEDDING_TOP_FEATURES_PER_SAMPLE
        )
        self.features_per_chunk = int(
            features_per_chunk
            if features_per_chunk is not None
            else EMBEDDING_FEATURES_PER_CHUNK
        )
        self.value_decimals = int(
            value_decimals
            if value_decimals is not None
            else EMBEDDING_VALUE_DECIMALS
        )
        self.sort_features_by_magnitude = bool(
            sort_features_by_magnitude
            if sort_features_by_magnitude is not None
            else EMBEDDING_SORT_FEATURES_BY_MAGNITUDE
        )
        self.enforce_max_length = bool(
            enforce_max_length
            if enforce_max_length is not None
            else EMBEDDING_ENFORCE_MAX_LENGTH
        )
        self.device = device if device is not None else EMBEDDING_DEVICE
        self.show_progress_bar = bool(
            show_progress_bar
            if show_progress_bar is not None
            else EMBEDDING_SHOW_PROGRESS
        )

        if not self.embedding_model_name:
            raise ValueError("embedding_model_name non può essere vuoto.")
        if self.embedding_batch_size <= 0:
            raise ValueError(
                "embedding_batch_size deve essere maggiore di zero."
            )
        if self.embedding_sample_batch_size <= 0:
            raise ValueError(
                "embedding_sample_batch_size deve essere maggiore di zero."
            )
        if self.top_features_per_sample <= 0:
            raise ValueError(
                "top_features_per_sample deve essere maggiore di zero."
            )
        if self.features_per_chunk <= 0:
            raise ValueError(
                "features_per_chunk deve essere maggiore di zero."
            )
        if self.value_decimals < 0:
            raise ValueError("value_decimals non può essere negativo.")
        if self.device not in {None, "cpu", "cuda"}:
            raise ValueError("device deve essere None, 'cpu' oppure 'cuda'.")

        self.global_index = None
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []
        self.empty_feature_names: list[str] = []
        self.labels: Optional[np.ndarray] = None
        self.z_vectors: Optional[np.ndarray] = None
        self.cache_signature: Optional[str] = None
        self.cached_semantic_config: dict[str, object] = {}

        self._embedding_model: Optional[Embedding] = None
        self._text_encoder: Optional[FeatureTextEncoder] = None

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS non è installato. Eseguire: pip install faiss-cpu"
            ) from exc
        return faiss

    @staticmethod
    def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
        result = np.asarray(vectors, dtype=np.float32).copy()
        if result.ndim != 2:
            raise ValueError("I vettori devono formare una matrice 2D.")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise RuntimeError(
                "Non è possibile normalizzare embedding non validi."
            )
        result /= norms
        return result

    def semantic_config(self) -> dict[str, object]:
        return {
            "embedding_model_name": self.embedding_model_name,
            "top_features_per_sample": self.top_features_per_sample,
            "features_per_chunk": self.features_per_chunk,
            "value_decimals": self.value_decimals,
            "sort_features_by_magnitude": self.sort_features_by_magnitude,
            "enforce_max_length": self.enforce_max_length,
        }

    def _ensure_embedding_model(self) -> Embedding:
        if self._embedding_model is None:
            self._embedding_model = Embedding(
                model_name=self.embedding_model_name,
                device=self.device,
            )
        return self._embedding_model

    def _build_text_encoder(self) -> FeatureTextEncoder:
        if not self.feature_names:
            raise RuntimeError(
                "I nomi delle feature non sono stati inizializzati."
            )
        self._text_encoder = FeatureTextEncoder(
            feature_names=self.feature_names,
            feature_descriptions=(
                BODMASFeatureNames.get_feature_descriptions()
            ),
            top_features_per_sample=self.top_features_per_sample,
            features_per_chunk=self.features_per_chunk,
            value_decimals=self.value_decimals,
            sort_by_magnitude=self.sort_features_by_magnitude,
        )
        return self._text_encoder

    def _get_text_encoder(self) -> FeatureTextEncoder:
        return self._text_encoder or self._build_text_encoder()

    def _encode_sample_batch(self, z_batch: np.ndarray) -> np.ndarray:
        model = self._ensure_embedding_model()
        text_encoder = self._get_text_encoder()

        chunks_by_sample = [
            text_encoder.row_to_chunks(row) for row in z_batch
        ]
        flat_chunks = [
            chunk
            for sample_chunks in chunks_by_sample
            for chunk in sample_chunks
        ]
        if not flat_chunks:
            raise RuntimeError("Nessun segmento da codificare.")

        chunk_embeddings = model.encode(
            flat_chunks,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            enforce_max_length=self.enforce_max_length,
        )

        sample_embeddings: list[np.ndarray] = []
        offset = 0
        for sample_chunks in chunks_by_sample:
            next_offset = offset + len(sample_chunks)
            current = chunk_embeddings[offset:next_offset]
            if len(current) != len(sample_chunks):
                raise RuntimeError(
                    "Numero di embedding dei segmenti non coerente."
                )
            sample_embeddings.append(current.mean(axis=0))
            offset = next_offset

        if offset != len(chunk_embeddings):
            raise RuntimeError(
                "Errore nell'aggregazione degli embedding dei segmenti."
            )
        return self._l2_normalise(
            np.vstack(sample_embeddings).astype(np.float32)
        )

    def _encode_z_matrix(
        self,
        z_matrix: np.ndarray,
        description: str,
    ) -> np.ndarray:
        matrix = np.asarray(z_matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("z_matrix deve essere una matrice 2D.")
        if len(matrix) == 0:
            raise ValueError("z_matrix non può essere vuota.")
        if not np.isfinite(matrix).all():
            raise ValueError("z_matrix contiene NaN o infiniti.")

        starts = range(0, len(matrix), self.embedding_sample_batch_size)
        iterator = tqdm(
            starts,
            total=(
                len(matrix) + self.embedding_sample_batch_size - 1
            ) // self.embedding_sample_batch_size,
            desc=description,
            unit="batch",
            disable=(
                not self.show_progress_bar
                or len(matrix) <= self.embedding_sample_batch_size
            ),
        )
        batches = [
            self._encode_sample_batch(
                matrix[start:start + self.embedding_sample_batch_size]
            )
            for start in iterator
        ]
        result = np.vstack(batches).astype(np.float32)
        if len(result) != len(matrix):
            raise RuntimeError(
                "Numero di embedding diverso dal numero di campioni."
            )
        return result

    def fit(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        feature_names: list[str],
        cache_signature: Optional[str] = None,
    ) -> None:
        faiss = self._faiss()
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if len(X) != len(labels):
            raise ValueError(
                "Numero di righe ed etichette non coincidente."
            )
        if X.empty:
            raise ValueError("Il training set è vuoto.")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("L'indice richiede entrambe le classi 0 e 1.")
        if not feature_names:
            raise ValueError("feature_names non può essere vuoto.")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names contiene duplicati.")

        missing = [name for name in feature_names if name not in X.columns]
        if missing:
            raise ValueError(
                f"Feature mancanti nel training set: {missing[:10]}"
            )

        self.feature_names = list(feature_names)
        self._embedding_model = None
        self._build_text_encoder()

        selected = X.loc[:, self.feature_names].copy()
        self.empty_feature_names = [
            name for name in self.feature_names
            if selected[name].isna().all()
        ]
        if self.empty_feature_names:
            selected.loc[:, self.empty_feature_names] = 0.0

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        imputed = self.imputer.fit_transform(selected)
        self.z_vectors = self.scaler.fit_transform(imputed).astype(np.float32)
        if self.z_vectors.shape != (len(X), len(self.feature_names)):
            raise RuntimeError(
                "Imputazione o standardizzazione ha cambiato la dimensione "
                "delle feature."
            )
        if not np.isfinite(self.z_vectors).all():
            raise RuntimeError(
                "Il training standardizzato contiene NaN o infiniti."
            )

        retrieval_vectors = self._encode_z_matrix(
            self.z_vectors,
            description="Embedding training",
        )
        self.global_index = faiss.IndexFlatIP(retrieval_vectors.shape[1])
        self.global_index.add(retrieval_vectors)
        self.labels = labels.copy()
        self.cache_signature = cache_signature
        self.cached_semantic_config = self.semantic_config()

    def _check_ready(self) -> None:
        if (
            self.global_index is None
            or self.imputer is None
            or self.scaler is None
            or self.labels is None
            or self.z_vectors is None
            or not self.feature_names
        ):
            raise RuntimeError("Indice FAISS non costruito o non caricato.")

    def _to_frame(
        self,
        rows: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
    ) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            missing = [
                name for name in self.feature_names
                if name not in rows.columns
            ]
            if missing:
                raise ValueError(
                    f"Feature mancanti nella query: {missing[:10]}"
                )
            frame = rows.loc[:, self.feature_names].copy()
        elif isinstance(rows, pd.Series):
            if any(name not in rows.index for name in self.feature_names):
                missing = [
                    name for name in self.feature_names
                    if name not in rows.index
                ]
                raise ValueError(
                    f"Feature mancanti nella query: {missing[:10]}"
                )
            frame = pd.DataFrame(
                [rows.reindex(self.feature_names).to_numpy()],
                columns=self.feature_names,
            )
        else:
            values = np.asarray(rows, dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.ndim != 2:
                raise ValueError(
                    "La query deve essere un vettore o una matrice."
                )
            if values.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"La query ha {values.shape[1]} feature; "
                    f"attese {len(self.feature_names)}."
                )
            frame = pd.DataFrame(values, columns=self.feature_names)

        frame = frame.apply(pd.to_numeric, errors="coerce")
        values = frame.to_numpy(dtype=np.float64, copy=True)
        values[np.isinf(values)] = np.nan
        frame = pd.DataFrame(values, columns=self.feature_names)
        if self.empty_feature_names:
            frame.loc[:, self.empty_feature_names] = frame.loc[
                :, self.empty_feature_names
            ].fillna(0.0)
        return frame

    def transform_queries(
        self,
        rows: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        frame = self._to_frame(rows)
        imputed = self.imputer.transform(frame)
        z_queries = self.scaler.transform(imputed).astype(np.float32)
        if not np.isfinite(z_queries).all():
            raise RuntimeError(
                "Le query standardizzate contengono NaN o infiniti."
            )
        retrieval_queries = self._encode_z_matrix(
            z_queries,
            description="Embedding test",
        )
        return z_queries, retrieval_queries

    def transform_query(
        self,
        row: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
    ) -> tuple[np.ndarray, np.ndarray]:
        z_queries, retrieval_queries = self.transform_queries(row)
        if len(z_queries) != 1:
            raise ValueError(
                "transform_query accetta un solo campione."
            )
        return z_queries, retrieval_queries

    def search_many(
        self,
        rows: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
    ) -> list[list[Neighbor]]:
        self._check_ready()
        k = int(k)
        if k <= 0:
            raise ValueError("k deve essere maggiore di zero.")
        if int(self.global_index.ntotal) <= 0:
            raise RuntimeError("L'indice FAISS è vuoto.")

        _, queries = self.transform_queries(rows)
        real_k = min(k, int(self.global_index.ntotal))
        similarities, indices = self.global_index.search(
            queries.astype(np.float32),
            real_k,
        )

        result: list[list[Neighbor]] = []
        for query_similarities, query_indices in zip(
            similarities,
            indices,
        ):
            neighbors = [
                Neighbor(
                    index=int(index),
                    label=int(self.labels[index]),
                    similarity=float(similarity),
                    z_vector=self.z_vectors[index].copy(),
                )
                for similarity, index in zip(
                    query_similarities,
                    query_indices,
                )
                if int(index) >= 0
            ]
            if not neighbors:
                raise RuntimeError("FAISS non ha restituito vicini validi.")
            result.append(neighbors)
        return result

    def search(
        self,
        row: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
    ) -> list[Neighbor]:
        results = self.search_many(row, k)
        if len(results) != 1:
            raise ValueError("search accetta un solo campione.")
        return results[0]

    @staticmethod
    def _similarity_weight(similarity: float) -> float:
        # Le similarità negative non devono fornire evidenza positiva.
        return max(float(similarity), 0.0)

    @classmethod
    def _vote_from_neighbors(
        cls,
        neighbors: list[Neighbor],
        strategy: str,
    ) -> VoteResult:
        if not neighbors:
            raise RuntimeError("FAISS non ha restituito vicini.")
        if strategy not in {"majority", "similarity_weighted"}:
            raise ValueError(
                "strategy deve essere 'majority' oppure "
                "'similarity_weighted'."
            )

        counts = {
            0: sum(neighbor.label == 0 for neighbor in neighbors),
            1: sum(neighbor.label == 1 for neighbor in neighbors),
        }
        scores = {
            0: float(sum(
                cls._similarity_weight(neighbor.similarity)
                for neighbor in neighbors if neighbor.label == 0
            )),
            1: float(sum(
                cls._similarity_weight(neighbor.similarity)
                for neighbor in neighbors if neighbor.label == 1
            )),
        }

        if strategy == "majority":
            if counts[0] != counts[1]:
                prediction = 0 if counts[0] > counts[1] else 1
            elif not np.isclose(scores[0], scores[1]):
                prediction = 0 if scores[0] > scores[1] else 1
            else:
                prediction = int(neighbors[0].label)
        else:
            if not np.isclose(scores[0], scores[1]):
                prediction = 0 if scores[0] > scores[1] else 1
            elif counts[0] != counts[1]:
                prediction = 0 if counts[0] > counts[1] else 1
            else:
                prediction = int(neighbors[0].label)

        return VoteResult(
            prediction=int(prediction),
            counts=counts,
            similarity_scores=scores,
            neighbors=neighbors,
            strategy=strategy,
        )

    def vote_many(
        self,
        rows: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
        strategy: str = "majority",
    ) -> list[VoteResult]:
        return [
            self._vote_from_neighbors(neighbors, strategy)
            for neighbors in self.search_many(rows, k)
        ]

    def vote(
        self,
        row: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
        strategy: str = "majority",
    ) -> VoteResult:
        neighbors = self.search(row, k)
        return self._vote_from_neighbors(neighbors, strategy)

    def majority_vote_many(
        self,
        rows: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
    ) -> list[VoteResult]:
        return self.vote_many(rows, k, strategy="majority")

    def majority_vote(
        self,
        row: pd.Series | pd.DataFrame | np.ndarray | list | tuple,
        k: int,
    ) -> VoteResult:
        return self.vote(row, k, strategy="majority")

    @staticmethod
    def _paths(prefix: str | os.PathLike[str]) -> dict[str, Path]:
        base = Path(prefix)
        return {
            "global": Path(str(base) + "_global.faiss"),
            "meta": Path(str(base) + "_metadata.pkl"),
        }

    @classmethod
    def exists(cls, prefix: str | os.PathLike[str]) -> bool:
        return all(path.exists() for path in cls._paths(prefix).values())

    def save(self, prefix: str | os.PathLike[str]) -> None:
        faiss = self._faiss()
        self._check_ready()
        paths = self._paths(prefix)
        paths["meta"].parent.mkdir(parents=True, exist_ok=True)

        global_tmp = Path(str(paths["global"]) + ".tmp")
        meta_tmp = Path(str(paths["meta"]) + ".tmp")
        try:
            faiss.write_index(self.global_index, str(global_tmp))
            with meta_tmp.open("wb") as handle:
                pickle.dump(
                    {
                        "cache_format_version": 2,
                        "feature_names": self.feature_names,
                        "empty_feature_names": self.empty_feature_names,
                        "labels": self.labels,
                        "z_vectors": self.z_vectors,
                        "imputer": self.imputer,
                        "scaler": self.scaler,
                        "cache_signature": self.cache_signature,
                        "semantic_config": self.semantic_config(),
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(global_tmp, paths["global"])
            os.replace(meta_tmp, paths["meta"])
        finally:
            for temporary in (global_tmp, meta_tmp):
                if temporary.exists():
                    temporary.unlink()

    def load(self, prefix: str | os.PathLike[str]) -> None:
        faiss = self._faiss()
        paths = self._paths(prefix)
        if not all(path.exists() for path in paths.values()):
            raise FileNotFoundError(f"Cache FAISS incompleta: {prefix!s}")

        loaded_index = faiss.read_index(str(paths["global"]))
        with paths["meta"].open("rb") as handle:
            metadata = pickle.load(handle)

        feature_names = list(metadata["feature_names"])
        labels = np.asarray(metadata["labels"], dtype=np.int64)
        z_vectors = np.asarray(metadata["z_vectors"], dtype=np.float32)
        imputer = metadata["imputer"]
        scaler = metadata["scaler"]

        if int(loaded_index.ntotal) != len(labels):
            raise RuntimeError(
                "La cache FAISS contiene un numero incoerente di vettori."
            )
        if z_vectors.ndim != 2 or len(z_vectors) != len(labels):
            raise RuntimeError("z_vectors della cache non coerenti.")
        if z_vectors.shape[1] != len(feature_names):
            raise RuntimeError(
                "Numero di feature della cache non coerente."
            )
        if not np.isfinite(z_vectors).all():
            raise RuntimeError("La cache contiene NaN o infiniti.")

        self.global_index = loaded_index
        self.feature_names = feature_names
        self.empty_feature_names = list(
            metadata.get("empty_feature_names", [])
        )
        self.labels = labels
        self.z_vectors = z_vectors
        self.imputer = imputer
        self.scaler = scaler
        self.cache_signature = metadata.get("cache_signature")
        self.cached_semantic_config = dict(
            metadata.get("semantic_config", {})
        )

        # Non sovrascrive batch size, device o configurazione desiderata con
        # valori obsoleti della cache. La firma decide se la cache è riusabile.
        self._embedding_model = None
        self._build_text_encoder()
