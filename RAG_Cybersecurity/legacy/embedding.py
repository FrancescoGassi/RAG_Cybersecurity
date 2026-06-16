from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch


class Embedding:
    """Wrapper validato per SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        if not model_name or not str(model_name).strip():
            raise ValueError("model_name non può essere vuoto.")
        if device not in {None, "cpu", "cuda"}:
            raise ValueError("device deve essere None, 'cpu' oppure 'cuda'.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "È stato richiesto CUDA, ma CUDA non è disponibile."
            )

        self.model_name = str(model_name).strip()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers non è installato. "
                "Eseguire: pip install sentence-transformers"
            ) from exc

        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        return SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                "Il modello non ha restituito la dimensione degli embedding."
            )
        return int(dimension)

    @property
    def max_sequence_length(self) -> int | None:
        value = getattr(self._model, "max_seq_length", None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _validate_token_lengths(self, texts: list[str]) -> None:
        tokenizer = getattr(self._model, "tokenizer", None)
        maximum = self.max_sequence_length
        if tokenizer is None or maximum is None:
            return

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded.get("input_ids", [])
        lengths = [len(ids) for ids in input_ids]
        if lengths and max(lengths) > maximum:
            longest = max(lengths)
            raise ValueError(
                "Un segmento testuale supera il limite del modello: "
                f"{longest} token, massimo {maximum}. Ridurre "
                "EMBEDDING_FEATURES_PER_CHUNK."
            )

    def encode(
        self,
        sentences: str | Sequence[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        enforce_max_length: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            texts = [sentences]
        else:
            texts = [str(sentence) for sentence in sentences]

        if not texts:
            raise ValueError("Non è stato fornito alcun testo da codificare.")
        if any(not text.strip() for text in texts):
            raise ValueError("I testi da codificare non possono essere vuoti.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size deve essere maggiore di zero.")

        if enforce_max_length:
            self._validate_token_lengths(texts)

        vectors = self._model.encode(
            texts,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            show_progress_bar=bool(show_progress_bar),
            normalize_embeddings=True,
        )
        result = np.asarray(vectors, dtype=np.float32)
        if result.ndim == 1:
            result = result.reshape(1, -1)
        if result.ndim != 2 or len(result) != len(texts):
            raise RuntimeError(
                "Il modello ha restituito embedding con forma non valida."
            )
        if not np.isfinite(result).all():
            raise RuntimeError("Gli embedding contengono NaN o infiniti.")
        return result
