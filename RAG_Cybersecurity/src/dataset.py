from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

_LABEL_MAP = {
    "0": 0, "goodware": 0, "benign": 0, "safe": 0, "clean": 0, "false": 0,
    "1": 1, "malware": 1, "malicious": 1, "threat": 1, "true": 1,
}

def normalise_binary_labels(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("La colonna target contiene valori mancanti.")

    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="raise")
        unique = set(pd.unique(numeric).tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Target numerico non binario: {sorted(unique)[:10]}")
        return numeric.astype(np.int64)

    mapped = values.astype(str).str.strip().str.lower().map(_LABEL_MAP)
    if mapped.isna().any():
        invalid = values[mapped.isna()].astype(str).unique().tolist()[:10]
        raise ValueError(f"Etichette non riconosciute: {invalid}")
    return mapped.astype(np.int64)

@dataclass
class Dataset:
    feat_data: pd.DataFrame
    target_data: pd.Series
    target_name: str

    @classmethod
    def from_csv(cls, file_path: str, target_column: Optional[str] = None) -> "Dataset":
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Il dataset {file_path!r} è vuoto.")

        target_name = target_column or str(df.columns[-1])
        if target_name not in df.columns:
            raise ValueError(f"Colonna target {target_name!r} non trovata.")

        y = normalise_binary_labels(df[target_name]).reset_index(drop=True)
        X = df.drop(columns=[target_name]).copy()
        if X.shape[1] == 0:
            raise ValueError("Il dataset non contiene feature.")

        X.columns = X.columns.astype(str)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        return cls(X, y, target_name)

    @property
    def feature_names(self) -> list[str]:
        return self.feat_data.columns.tolist()

    def __len__(self) -> int:
        return len(self.target_data)

    def class_counts(self) -> dict[int, int]:
        counts = self.target_data.value_counts().sort_index()
        return {int(label): int(count) for label, count in counts.items()}

    def subset(self, indices: Iterable[int]) -> "Dataset":
        idx = np.asarray(list(indices), dtype=np.int64)
        return Dataset(
            feat_data=self.feat_data.iloc[idx].reset_index(drop=True),
            target_data=self.target_data.iloc[idx].reset_index(drop=True),
            target_name=self.target_name,
        )

    def stratified_sample(self, limit: Optional[int], random_state: int = 42, balanced: bool = False) -> "Dataset":
        if self.target_data.nunique() != 2:
            raise ValueError(f"Sono necessarie entrambe le classi; distribuzione: {self.class_counts()}")
        if limit is None or limit >= len(self):
            if not balanced:
                return self.subset(range(len(self)))
            limit = len(self)
        if limit < 2:
            raise ValueError("Il campione deve contenere almeno due righe.")

        rng = np.random.default_rng(random_state)
        class_indices = {label: np.flatnonzero(self.target_data.to_numpy() == label) for label in (0, 1)}

        if balanced:
            per_class = min(limit // 2, len(class_indices[0]), len(class_indices[1]))
            if per_class == 0:
                raise ValueError("Impossibile creare un campione bilanciato.")
            selected: list[int] = []
            for label in (0, 1):
                selected.extend(rng.choice(class_indices[label], size=per_class, replace=False).tolist())
            if limit % 2 == 1 and len(selected) < limit:
                remaining = np.setdiff1d(np.arange(len(self)), np.asarray(selected))
                if len(remaining):
                    selected.append(int(rng.choice(remaining)))
        else:
            count0, count1 = len(class_indices[0]), len(class_indices[1])
            n0 = int(round(limit * count0 / (count0 + count1)))
            n0 = max(1, min(n0, count0, limit - 1))
            n1 = max(1, min(limit - n0, count1))
            selected = rng.choice(class_indices[0], size=n0, replace=False).tolist()
            selected += rng.choice(class_indices[1], size=n1, replace=False).tolist()

        rng.shuffle(selected)
        return self.subset(selected)

    def compute_mutual_information(self, random_state: int = 42) -> dict[str, float]:
        if self.target_data.nunique() != 2:
            raise ValueError("La Mutual Information richiede entrambe le classi.")
        X = self.feat_data.copy()
        medians = X.median(axis=0, skipna=True).fillna(0.0)
        X = X.fillna(medians).astype(np.float32)
        scores = mutual_info_classif(X, self.target_data.to_numpy(dtype=np.int64), random_state=random_state)
        ordered = sorted(zip(self.feature_names, scores.tolist()), key=lambda item: item[1], reverse=True)
        return {name: float(score) for name, score in ordered}

    def select_features(self, feature_names: list[str]) -> "Dataset":
        missing = [name for name in feature_names if name not in self.feat_data.columns]
        if missing:
            raise ValueError(f"Feature mancanti nel dataset: {missing[:10]}")
        return Dataset(
            feat_data=self.feat_data.loc[:, feature_names].copy(),
            target_data=self.target_data.copy(),
            target_name=self.target_name,
        )