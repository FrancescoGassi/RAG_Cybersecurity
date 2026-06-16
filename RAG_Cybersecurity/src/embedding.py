from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class EmbeddingModel:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        if not model_name.strip():
            raise ValueError("Il nome del modello non può essere vuoto.")
        if device not in {None, "cpu", "cuda"}:
            raise ValueError("device deve essere None, 'cpu' oppure 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA è stata richiesta ma non è disponibile.")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name.strip()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else [str(x) for x in texts]
        if not values or any(not value.strip() for value in values):
            raise ValueError("I testi da codificare non possono essere vuoti.")
        if batch_size <= 0:
            raise ValueError("batch_size deve essere maggiore di zero.")

        vectors = self.model.encode(
            values,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or len(result) != len(values):
            raise RuntimeError("Il modello ha restituito embedding non validi.")
        if not np.isfinite(result).all():
            raise RuntimeError("Gli embedding contengono NaN o infiniti.")
        return result
