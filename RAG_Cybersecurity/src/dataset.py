from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


_LABEL_MAP = {
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


def normalise_binary_labels(values: pd.Series) -> pd.Series:
    """Converte le etichette in 0=goodware e 1=malware."""
    if values.isna().any():
        raise ValueError("La colonna target contiene valori mancanti.")

    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="raise")
        unique = set(pd.unique(numeric).tolist())

        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"Target numerico non binario: {sorted(unique)[:10]}"
            )

        return numeric.astype(np.int64)

    mapped = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map(_LABEL_MAP)
    )

    if mapped.isna().any():
        invalid = (
            values[mapped.isna()]
            .astype(str)
            .unique()
            .tolist()[:10]
        )
        raise ValueError(f"Etichette non riconosciute: {invalid}")

    return mapped.astype(np.int64)


@dataclass
class Dataset:
    feat_data: pd.DataFrame
    target_data: pd.Series
    target_name: str

    @staticmethod
    def _get_target_name(
        file_path: str,
        target_column: Optional[str],
    ) -> str:
        header = pd.read_csv(file_path, nrows=0)

        if len(header.columns) < 2:
            raise ValueError(
                f"Il dataset {file_path!r} deve contenere feature e target."
            )

        target_name = target_column or str(header.columns[-1])

        if target_name not in header.columns:
            raise ValueError(
                f"Colonna target {target_name!r} non trovata."
            )

        return target_name

    @staticmethod
    def _select_indices(
        labels: pd.Series,
        limit: Optional[int],
        random_state: int,
        balanced: bool,
    ) -> np.ndarray:
        y = labels.to_numpy(dtype=np.int64)
        total = len(y)

        if total == 0:
            raise ValueError("Il dataset non contiene righe.")

        if set(np.unique(y).tolist()) != {0, 1}:
            raise ValueError(
                "Sono necessarie entrambe le classi 0 e 1."
            )

        class_indices = {
            0: np.flatnonzero(y == 0),
            1: np.flatnonzero(y == 1),
        }

        if limit is None:
            if not balanced:
                return np.arange(total, dtype=np.int64)
            limit = 2 * min(
                len(class_indices[0]),
                len(class_indices[1]),
            )

        if limit < 2:
            raise ValueError(
                "Il campione deve contenere almeno due righe."
            )

        limit = min(int(limit), total)

        if limit == total and not balanced:
            return np.arange(total, dtype=np.int64)

        rng = np.random.default_rng(random_state)
        selected: list[int] = []

        if balanced:
            per_class = min(
                limit // 2,
                len(class_indices[0]),
                len(class_indices[1]),
            )

            if per_class == 0:
                raise ValueError(
                    "Impossibile creare un campione bilanciato."
                )

            for label in (0, 1):
                chosen = rng.choice(
                    class_indices[label],
                    size=per_class,
                    replace=False,
                )
                selected.extend(chosen.tolist())

            if limit % 2 == 1 and len(selected) < limit:
                remaining = np.setdiff1d(
                    np.arange(total),
                    np.asarray(selected, dtype=np.int64),
                )
                if len(remaining) > 0:
                    selected.append(int(rng.choice(remaining)))
        else:
            count0 = len(class_indices[0])
            count1 = len(class_indices[1])

            n0 = int(round(limit * count0 / (count0 + count1)))
            n0 = max(1, min(n0, count0, limit - 1))
            n1 = limit - n0
            n1 = max(1, min(n1, count1))

            selected.extend(
                rng.choice(
                    class_indices[0],
                    size=n0,
                    replace=False,
                ).tolist()
            )
            selected.extend(
                rng.choice(
                    class_indices[1],
                    size=n1,
                    replace=False,
                ).tolist()
            )

        rng.shuffle(selected)
        return np.asarray(selected, dtype=np.int64)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        target_column: Optional[str] = None,
        limit: Optional[int] = None,
        random_state: int = 42,
        balanced: bool = False,
    ) -> "Dataset":
        """
        Legge prima soltanto il target, seleziona gli indici e poi carica
        esclusivamente le righe necessarie. Evita il caricamento completo
        di matrici molto grandi.
        """
        target_name = cls._get_target_name(
            file_path,
            target_column,
        )

        target_frame = pd.read_csv(
            file_path,
            usecols=[target_name],
        )
        all_labels = normalise_binary_labels(
            target_frame[target_name]
        ).reset_index(drop=True)

        selected_indices = cls._select_indices(
            labels=all_labels,
            limit=limit,
            random_state=random_state,
            balanced=balanced,
        )

        if len(selected_indices) == len(all_labels):
            df = pd.read_csv(
                file_path,
                low_memory=False,
            )
        else:
            selected_sorted = np.sort(selected_indices)
            selected_csv_lines = {
                int(index) + 1
                for index in selected_sorted
            }

            df = pd.read_csv(
                file_path,
                skiprows=lambda line_number: (
                    line_number != 0
                    and line_number not in selected_csv_lines
                ),
                low_memory=False,
            )

            if len(df) != len(selected_sorted):
                raise RuntimeError(
                    "Il numero di righe caricate non coincide "
                    "con il campione selezionato."
                )

            # read_csv restituisce le righe nell'ordine originale del file.
            # Si ripristina l'ordine casuale prodotto da _select_indices().
            df.insert(
                0,
                "__source_row_index__",
                selected_sorted,
            )
            df = (
                df.set_index("__source_row_index__")
                .loc[selected_indices]
                .reset_index(drop=True)
            )

        if df.empty:
            raise ValueError(
                f"Nessuna riga caricata da {file_path!r}."
            )

        y = normalise_binary_labels(
            df[target_name]
        ).reset_index(drop=True)

        X = df.drop(columns=[target_name]).copy()

        if X.shape[1] == 0:
            raise ValueError(
                "Il dataset non contiene feature."
            )

        X.columns = X.columns.astype(str)
        X = X.apply(pd.to_numeric, errors="coerce")

        values = X.to_numpy(
            dtype=np.float32,
            copy=True,
        )
        infinite_mask = np.isinf(values)

        if infinite_mask.any():
            values[infinite_mask] = np.nan

        X = pd.DataFrame(
            values,
            columns=X.columns,
        ).reset_index(drop=True)

        return cls(
            feat_data=X,
            target_data=y,
            target_name=target_name,
        )

    @property
    def feature_names(self) -> list[str]:
        return self.feat_data.columns.tolist()

    def __len__(self) -> int:
        return len(self.target_data)

    def class_counts(self) -> dict[int, int]:
        counts = self.target_data.value_counts().sort_index()
        return {
            int(label): int(count)
            for label, count in counts.items()
        }

    def subset(
        self,
        indices: Iterable[int],
    ) -> "Dataset":
        idx = np.asarray(
            list(indices),
            dtype=np.int64,
        )

        return Dataset(
            feat_data=self.feat_data.iloc[idx].reset_index(drop=True),
            target_data=self.target_data.iloc[idx].reset_index(drop=True),
            target_name=self.target_name,
        )

    def compute_mutual_information(
        self,
        random_state: int = 42,
    ) -> dict[str, float]:
        if self.target_data.nunique() != 2:
            raise ValueError(
                "La Mutual Information richiede entrambe le classi."
            )

        X = self.feat_data.copy()
        medians = (
            X.median(axis=0, skipna=True)
            .fillna(0.0)
        )
        X = X.fillna(medians).astype(np.float32)

        scores = mutual_info_classif(
            X,
            self.target_data.to_numpy(dtype=np.int64),
            random_state=random_state,
        )

        ordered = sorted(
            zip(self.feature_names, scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            name: float(score)
            for name, score in ordered
        }

    def select_features(
        self,
        feature_names: list[str],
    ) -> "Dataset":
        missing = [
            name
            for name in feature_names
            if name not in self.feat_data.columns
        ]

        if missing:
            raise ValueError(
                f"Feature mancanti nel dataset: {missing[:10]}"
            )

        return Dataset(
            feat_data=self.feat_data.loc[
                :,
                feature_names,
            ].copy(),
            target_data=self.target_data.copy(),
            target_name=self.target_name,
        )