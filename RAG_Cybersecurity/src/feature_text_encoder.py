from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


TEXT_ENCODING_VERSION = "feature-text-v2-fixed-order"


@dataclass(frozen=True)
class FeatureTextEncoder:
    """Converte vettori standardizzati in testi privi di etichetta."""

    feature_names: Sequence[str]
    feature_descriptions: Mapping[str, str]
    top_features_per_sample: int = 32
    features_per_chunk: int = 8
    value_decimals: int = 4
    sort_by_magnitude: bool = False

    def __post_init__(self) -> None:
        names = [str(name) for name in self.feature_names]
        if not names:
            raise ValueError("feature_names non può essere vuoto.")
        if len(set(names)) != len(names):
            raise ValueError("feature_names contiene duplicati.")
        if int(self.top_features_per_sample) <= 0:
            raise ValueError(
                "top_features_per_sample deve essere maggiore di zero."
            )
        if int(self.features_per_chunk) <= 0:
            raise ValueError(
                "features_per_chunk deve essere maggiore di zero."
            )
        if int(self.value_decimals) < 0:
            raise ValueError("value_decimals non può essere negativo.")

    @staticmethod
    def _qualitative_level(value: float) -> str:
        if value >= 2.0:
            return "very high"
        if value >= 1.0:
            return "high"
        if value >= 0.25:
            return "above average"
        if value > -0.25:
            return "near average"
        if value > -1.0:
            return "below average"
        if value > -2.0:
            return "low"
        return "very low"

    def _description(self, feature_name: str) -> str:
        return self.feature_descriptions.get(
            feature_name,
            feature_name.replace("_", " "),
        )

    def _selected_indices(self, row: np.ndarray) -> list[int]:
        count = min(int(self.top_features_per_sample), len(row))
        if count == len(row):
            selected = np.arange(len(row), dtype=np.int64)
        else:
            selected = np.argpartition(
                np.abs(row),
                -count,
            )[-count:]

        indices = [int(index) for index in selected]
        if self.sort_by_magnitude:
            return sorted(
                indices,
                key=lambda index: (-abs(float(row[index])), index),
            )

        # Ordine globale stabile: evita che lo stesso insieme di feature venga
        # presentato al modello in sequenze diverse tra campioni.
        return sorted(indices)

    def row_to_chunks(self, row: np.ndarray) -> list[str]:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"Il vettore contiene {len(values)} valori; "
                f"ne erano attesi {len(self.feature_names)}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                "Il vettore standardizzato contiene NaN o infiniti."
            )

        selected_indices = self._selected_indices(values)
        statements: list[str] = []
        for index in selected_indices:
            feature_name = str(self.feature_names[index])
            value = float(values[index])
            description = self._description(feature_name)
            level = self._qualitative_level(value)
            statements.append(
                f"{feature_name}: {description}; "
                f"standardized value={value:+.{self.value_decimals}f}; "
                f"level={level}"
            )

        chunk_size = int(self.features_per_chunk)
        statement_chunks = [
            statements[start:start + chunk_size]
            for start in range(0, len(statements), chunk_size)
        ]
        if not statement_chunks:
            raise RuntimeError("Nessun segmento testuale generato.")

        total_parts = len(statement_chunks)
        return [
            (
                "Windows PE static-analysis feature representation. "
                f"Part {part_number} of {total_parts}. "
                + " | ".join(chunk)
            )
            for part_number, chunk in enumerate(statement_chunks, start=1)
        ]
