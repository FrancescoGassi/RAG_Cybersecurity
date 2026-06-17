from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureTextEncoder:
    feature_names: list[str]
    top_features: int
    features_per_chunk: int

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError("feature_names non può essere vuoto.")
        if self.top_features <= 0 or self.features_per_chunk <= 0:
            raise ValueError("I parametri dell'encoder devono essere positivi.")

    def row_to_chunks(self, row: np.ndarray) -> list[str]:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if len(values) != len(self.feature_names):
            raise ValueError("Numero di valori diverso dal numero di feature.")
        if not np.isfinite(values).all():
            raise ValueError("Il vettore contiene NaN o infiniti.")

        count = min(self.top_features, len(values))
        if count == len(values):
            selected = np.arange(len(values))
        else:
            selected = np.argpartition(np.abs(values), -count)[-count:]

        selected = sorted(int(index) for index in selected)
        statements = [
            f"{self.feature_names[index]} has standardized value {values[index]:+.4f}"
            for index in selected
        ]

        chunks: list[str] = []
        for start in range(0, len(statements), self.features_per_chunk):
            part = statements[start:start + self.features_per_chunk]
            chunks.append(
                "Windows PE static features. " + "; ".join(part) + "."
            )
        return chunks