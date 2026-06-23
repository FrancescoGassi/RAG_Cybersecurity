from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


_LABELS = {
    "0": 0,
    "goodware": 0,
    "benign": 0,
    "safe": 0,
    "clean": 0,
    "false": 0,
    "1": 1,
    "malware": 1,
    "malicious": 1,
    "threat": 1,
    "true": 1,
}


def normalise_labels(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("La colonna target contiene valori mancanti.")

    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_numeric(values, errors="raise").astype(np.int64)
    else:
        result = (
            values.astype(str)
            .str.strip()
            .str.lower()
            .map(_LABELS)
        )
        if result.isna().any():
            invalid = values[result.isna()].astype(str).unique()[:10]
            raise ValueError(f"Etichette non riconosciute: {invalid.tolist()}")
        result = result.astype(np.int64)

    unique = set(result.unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Target non binario: {sorted(unique)}")
    return result.reset_index(drop=True)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series

    @classmethod
    def from_csv(
        cls,
        path: str,
        target_column: str | None = None,
        limit: int | None = None,
        random_seed: int = 42,
        balanced: bool = False,
    ) -> "Dataset":
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty or frame.shape[1] < 2:
            raise ValueError(f"Dataset non valido: {path}")

        target = target_column or str(frame.columns[-1])
        if target not in frame.columns:
            raise ValueError(f"Colonna target {target!r} non trovata in {path}")

        y = normalise_labels(frame[target])
        X = frame.drop(columns=[target]).copy()
        X.columns = X.columns.astype(str)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        if X.columns.duplicated().any():
            raise ValueError("Il dataset contiene nomi di feature duplicati.")
        if set(y.unique().tolist()) != {0, 1}:
            raise ValueError("Il dataset deve contenere entrambe le classi 0 e 1.")

        data = cls(X.reset_index(drop=True), y)
        return data.sample(limit, random_seed, balanced)

    def sample(
        self,
        limit: int | None,
        random_seed: int,
        balanced: bool,
    ) -> "Dataset":
        if limit is None and not balanced:
            return self

        rng = np.random.default_rng(random_seed)
        idx0 = np.flatnonzero(self.y.to_numpy() == 0)
        idx1 = np.flatnonzero(self.y.to_numpy() == 1)

        if balanced:
            per_class = min(len(idx0), len(idx1))
            if limit is not None:
                per_class = min(per_class, int(limit) // 2)
            if per_class <= 0:
                raise ValueError("Impossibile creare un campione bilanciato.")
            selected = np.concatenate([
                rng.choice(idx0, per_class, replace=False),
                rng.choice(idx1, per_class, replace=False),
            ])
        else:
            size = min(int(limit), len(self))
            if size < 2:
                raise ValueError("Il campione deve contenere almeno due righe.")
            selected = rng.choice(len(self), size, replace=False)

        rng.shuffle(selected)
        sampled = Dataset(
            self.X.iloc[selected].reset_index(drop=True),
            self.y.iloc[selected].reset_index(drop=True),
        )
        if set(sampled.y.unique().tolist()) != {0, 1}:
            raise ValueError("Il campione selezionato non contiene entrambe le classi.")
        return sampled

    def compute_mutual_info(self, random_seed: int) -> dict[str, float]:
        """
        Calcola il punteggio di mutual information per ogni feature del dataset.
        Gestisce valori mancanti e infiniti.
        """
        medians = self.X.median(numeric_only=True).fillna(0.0)
        X_ready = self.X.fillna(medians).fillna(0.0)

        scores = mutual_info_classif(
            X_ready.to_numpy(dtype=np.float32),
            self.y.to_numpy(dtype=np.int64),
            random_state=random_seed,
        )
        return {col: float(score) for col, score in zip(self.X.columns, scores)}

    def select_best_features(
        self,
        top_k: int | None,
        random_seed: int,
        precomputed_scores: dict[str, float] | None = None,
    ) -> tuple["Dataset", list[str], dict[str, float]]:
        """
        Seleziona le top_k feature in base alla mutual information.
        Se viene fornito precomputed_scores, lo usa senza ricalcolare.
        """
        if precomputed_scores is None:
            score_map = self.compute_mutual_info(random_seed)
        else:
            score_map = precomputed_scores

        if top_k is None or top_k >= len(self.X.columns):
            feature_names = self.X.columns.tolist()
            return Dataset(self.X.copy(), self.y.copy()), feature_names, score_map

        if top_k <= 0:
            raise ValueError("TOP_K_FEATURES deve essere maggiore di zero.")

        ranking = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))
        selected = [name for name, _ in ranking[:min(top_k, len(ranking))]]
        return Dataset(self.X[selected].copy(), self.y.copy()), selected, score_map

    def select_features(self, feature_names: list[str]) -> "Dataset":
        missing = [name for name in feature_names if name not in self.X.columns]
        if missing:
            raise ValueError(f"Feature mancanti: {missing[:10]}")
        return Dataset(self.X[feature_names].copy(), self.y.copy())

    def class_counts(self) -> dict[int, int]:
        counts = self.y.value_counts().sort_index()
        return {int(label): int(count) for label, count in counts.items()}

    def __len__(self) -> int:
        return len(self.y)