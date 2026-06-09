import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional
import logging

from text_dataset import TextDataset

logging.getLogger(__name__).setLevel(logging.ERROR)

class Dataset:
    def __init__(self, file_path: str):
        self.feat_data = None
        self.target_data = None
        self.feature_names = None
        self.target_name = None

        df = pd.read_csv(file_path)
        self.target_name = df.columns[-1]
        self.target_data = df[self.target_name].copy()
        self.feat_data = df.drop(columns=[self.target_name])
        self.feature_names = list(self.feat_data.columns)

    def compute_mutual_information(self, sample_size: Optional[int] = None) -> Dict[str, float]:
        """
        Calcola la Mutual Information. Se sample_size è specificato (es. 40000),
        utilizza un campione casuale per ridurre la memoria.
        Converte i dati in float32 per dimezzare l'occupazione.
        """
        if self.feat_data is None:
            raise ValueError("Dataset non caricato.")
        
        # Se richiesto, usa un sottocampione per il calcolo della MI
        if sample_size is not None and len(self.feat_data) > sample_size:
            sampled_idx = self.feat_data.sample(n=sample_size, random_state=42).index
            X = self.feat_data.loc[sampled_idx].astype(np.float32)
            y = self.target_data.loc[sampled_idx]
        else:
            X = self.feat_data.astype(np.float32)
            y = self.target_data

        mi = mutual_info_classif(X, y, random_state=42, n_jobs=-1)
        mi_dict = dict(zip(self.feature_names, mi))
        return dict(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))

    def sort_features_by_mi(self, mi_dict: Dict[str, float], top_k: Optional[int] = None) -> 'Dataset':
        if self.feat_data is None:
            raise ValueError("Dataset non caricato.")
        sorted_features = list(mi_dict.keys())
        if top_k is not None and top_k > 0:
            sorted_features = sorted_features[:top_k]
        new_dataset = Dataset.__new__(Dataset)
        new_dataset.target_name = self.target_name
        new_dataset.feature_names = sorted_features
        new_dataset.feat_data = self.feat_data[sorted_features].copy()
        new_dataset.target_data = self.target_data.copy()
        return new_dataset

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple['Dataset', 'Dataset']:
        X_train, X_test, y_train, y_test = train_test_split(
            self.feat_data, self.target_data,
            test_size=test_size, random_state=random_state, stratify=self.target_data
        )
        train_ds = Dataset.__new__(Dataset)
        train_ds.target_name = self.target_name
        train_ds.feature_names = self.feature_names
        train_ds.feat_data = X_train
        train_ds.target_data = y_train
        test_ds = Dataset.__new__(Dataset)
        test_ds.target_name = self.target_name
        test_ds.feature_names = self.feature_names
        test_ds.feat_data = X_test
        test_ds.target_data = y_test
        return train_ds, test_ds

    def _row_to_text(self, row_index: int, separator: str = ": ") -> str:
        if row_index < 0 or row_index >= len(self.feat_data):
            raise ValueError(f"Indice {row_index} non valido.")
        row = self.feat_data.iloc[row_index]
        parts = []
        for feat in self.feature_names:
            value = row[feat]
            if isinstance(value, float):
                if value == int(value):
                    value_str = str(int(value))
                else:
                    value_str = f"{value:.2f}"
            elif isinstance(value, (int, np.integer)):
                value_str = str(int(value))
            else:
                value_str = str(value)
            parts.append(f"{feat}{separator}{value_str}")
        return ", ".join(parts)

    def save(self, path: str):
        texts = [self._row_to_text(i) for i in range(len(self))]
        with open(path, 'wb') as f:
            pickle.dump((texts, self.target_data), f)

    def text_to_dataset(self) -> TextDataset:
        texts = [self._row_to_text(i) for i in range(len(self))]
        return TextDataset.from_data(texts, self.target_data)

    def __len__(self) -> int:
        return len(self.feat_data) if self.feat_data is not None else 0