import faiss
import numpy as np
import pickle
import os
from typing import Tuple, List, Union, Optional

from embedding_dataset import EmbeddingDataset

class VectorIndex:
    """
    Gestisce un indice vettoriale FAISS per ricerca di similarità.
    Attributi privati:
        _index : faiss.Index – l’indice FAISS (FlatL2 o FlatIP)
        _dimension : int – dimensionalità dei vettori
        _metadata : dict – metadati (targets, testi originali, num_vectors, index_type)
        _index_type : str – tipo di indice usato ("L2" o "IP")
    """

    def __init__(self, dimension: int = None):
        self._index = None
        self._dimension = dimension
        self._metadata = {}
        self._index_type = None

    def build(self, embedding_dataset: EmbeddingDataset, texts: List[str], metric: str = "L2"):
        embeddings = embedding_dataset.get_embedding()
        self._dimension = embeddings.shape[1]
        self._index_type = metric

        if metric == "L2":
            self._index = faiss.IndexFlatL2(self._dimension)
        elif metric == "IP":
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            raise ValueError(f"Metrica {metric} non supportata. Usare 'L2' o 'IP'.")

        self._index.add(embeddings.astype(np.float32))

        self._metadata['targets'] = embedding_dataset.get_target().tolist()
        self._metadata['texts'] = texts
        self._metadata['num_vectors'] = len(embeddings)
        self._metadata['index_type'] = self._index_type

    def save(self, path: str):
        if self._index is None:
            raise ValueError("Indice non costruito. Chiamare build() prima di save().")

        base_path = os.path.splitext(path)[0]
        index_path = base_path + ".faiss"
        meta_path = base_path + "_metadata.pkl"

        faiss.write_index(self._index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self._metadata, f)
        print(f"Indice salvato in {index_path}, metadati in {meta_path}")

    def load(self, path: str):
        base_path = os.path.splitext(path)[0]
        index_path = base_path + ".faiss"
        meta_path = base_path + "_metadata.pkl"

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"File indice {index_path} non trovato.")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"File metadati {meta_path} non trovato.")

        self._index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self._metadata = pickle.load(f)
        self._dimension = self._index.d
        self._index_type = self._metadata.get('index_type')
        print(f"Indice caricato da {index_path} (dimensione {self._dimension})")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise ValueError("Indice non caricato o non costruito.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        distances, indices = self._index.search(query_embedding, k)
        return distances[0], indices[0]

    def get_metadata(self, indices: Optional[Union[List[int], np.ndarray]] = None):
        if indices is None:
            return self._metadata
        return {
            'targets': [self._metadata['targets'][i] for i in indices],
            'texts': [self._metadata['texts'][i] for i in indices]
        }

    def get_metadata_by_indices(self, indices: Union[List[int], np.ndarray]) -> List[Tuple[str, int]]:
        if indices is None:
            raise ValueError("Indices cannot be None")
        return [(self._metadata['texts'][i], self._metadata['targets'][i]) for i in indices]

    def get_index_type(self) -> str:
        return self._index_type