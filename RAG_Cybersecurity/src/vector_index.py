import faiss
import numpy as np
import pickle
import os
from typing import Tuple, List, Union
from src.embedding_dataset import EmbeddingDataset
from src.text_dataset import TextDataset
from src.embedding import Embedding

class VectorIndex:
    def __init__(self, dimension: int = None):
        self._index = None
        self._dimension = dimension
        self._metadata = {}
        self._index_type = None

    def build_from_texts(self, text_dataset: TextDataset, emb_model: Embedding,
                         metric: str = "L2", batch_size: int = 64):
        """
        Costruisce l'indice FAISS direttamente dai testi, processando a lotti.
        Usa la metrica L2 (distanza euclidea).
        """
        texts = text_dataset.get_texts()
        targets = text_dataset.get_targets().tolist()
        if len(texts) == 0:
            raise ValueError("Nessun testo nel dataset")

        first_batch = texts[:batch_size]
        first_embs = emb_model.encode(first_batch, batch_size=batch_size)
        self._dimension = first_embs.shape[1]
        self._index_type = metric

        if metric == "L2":
            self._index = faiss.IndexFlatL2(self._dimension)
        elif metric == "IP":
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            raise ValueError(f"Metrica {metric} non supportata.")

        self._index.add(first_embs.astype(np.float32))

        for i in range(batch_size, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embs = emb_model.encode(batch, batch_size=batch_size)
            self._index.add(embs.astype(np.float32))

        self._metadata['targets'] = targets
        self._metadata['texts'] = texts
        self._metadata['num_vectors'] = len(texts)
        self._metadata['index_type'] = self._index_type

    def build(self, embedding_dataset: EmbeddingDataset, texts: List[str], metric: str = "L2"):
        embeddings = embedding_dataset.get_embedding()
        self._dimension = embeddings.shape[1]
        self._index_type = metric
        if metric == "L2":
            self._index = faiss.IndexFlatL2(self._dimension)
        elif metric == "IP":
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            raise ValueError(f"Metrica {metric} non supportata.")
        self._index.add(embeddings.astype(np.float32))
        self._metadata['targets'] = embedding_dataset.get_target().tolist()
        self._metadata['texts'] = texts
        self._metadata['num_vectors'] = len(embeddings)
        self._metadata['index_type'] = self._index_type

    def save(self, path: str):
        if self._index is None:
            raise ValueError("Indice non costruito.")
        base_path = os.path.splitext(path)[0]
        faiss.write_index(self._index, base_path + ".faiss")
        with open(base_path + "_metadata.pkl", 'wb') as f:
            pickle.dump(self._metadata, f)

    def load(self, path: str):
        base_path = os.path.splitext(path)[0]
        self._index = faiss.read_index(base_path + ".faiss")
        with open(base_path + "_metadata.pkl", 'rb') as f:
            self._metadata = pickle.load(f)
        self._dimension = self._index.d
        self._index_type = self._metadata.get('index_type')

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise ValueError("Indice non caricato.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self._index.search(query_embedding.astype(np.float32), k)
        return distances[0], indices[0]

    def get_metadata_by_indices(self, indices: Union[List[int], np.ndarray]) -> Tuple[List[str], List[int]]:
        if indices is None:
            raise ValueError("Indices cannot be None")
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        texts = [self._metadata['texts'][i] for i in indices]
        targets = [self._metadata['targets'][i] for i in indices]
        return texts, targets

    def get_index_type(self) -> str:
        return self._index_type