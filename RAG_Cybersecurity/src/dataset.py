import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataset:
    """
    Modella un dataset di file Windows PE (feature + target).
    Attributi:
        feat_data (pd.DataFrame): DataFrame con le 2381 colonne feature.
        target_data (pd.Series): Series con la colonna target (label 0/1).
        feature_names (List[str]): nomi simbolici delle feature, in ordine.
        target_name (str): nome dell'ultima colonna del CSV (target).
    """

    def __init__(self, file_path: str):
        """
        Carica il dataset dal file CSV.
        Il target_name viene ricavato automaticamente dall'ultima colonna.
        file_path non viene conservato come attributo.
        Args:
            file_path: percorso del CSV con intestazioni simboliche
        """
        self.feat_data = None
        self.target_data = None
        self.feature_names = None
        self.target_name = None

        try:
            df = pd.read_csv(file_path)

            self.target_name = df.columns[-1]
            self.target_data = df[self.target_name].copy()
            self.feat_data = df.drop(columns=[self.target_name])
            self.feature_names = list(self.feat_data.columns)

            logger.info(f"Dataset caricato: {len(self.feat_data)} esempi, {len(self.feature_names)} feature")

        except Exception as e:
            logger.error(f"Errore caricamento {file_path}: {e}")
            raise

    def compute_mutual_information(self) -> Dict[str, float]:
        """
        Calcola la mutua informazione tra ogni feature e il target.
        Restituisce un dizionario {feature: punteggio} ordinato decrescente.
        """
        if self.feat_data is None:
            raise ValueError("Dataset non caricato.")
        mi = mutual_info_classif(self.feat_data, self.target_data,
                                 random_state=42, n_jobs=-1)
        mi_dict = dict(zip(self.feature_names, mi))
        return dict(sorted(mi_dict.items(), key=lambda x: x[1], reverse=True))

    def sort_features_by_mi(self, mi_dict: Dict[str, float]) -> 'Dataset':
        """
        Richiama il dizionario {str: float} e restituisce un nuovo Dataset
        con le colonne riordinate per importanza decrescente.
        Args:
            mi_dict: dizionario {feature: punteggio} ordinato decrescente
        """
        if self.feat_data is None:
            raise ValueError("Dataset non caricato.")
        sorted_features = list(mi_dict.keys())

        new_dataset = Dataset.__new__(Dataset)
        new_dataset.target_name = self.target_name
        new_dataset.feature_names = sorted_features
        new_dataset.feat_data = self.feat_data[sorted_features].copy()
        new_dataset.target_data = self.target_data.copy()
        return new_dataset

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple['Dataset', 'Dataset']:
        """Divide i dati in training e test (stratificato sul target)."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.feat_data, self.target_data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.target_data
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

        logger.info(f"Split completato: train={len(train_ds)}, test={len(test_ds)}")
        return train_ds, test_ds

    def _row_to_text(self, row_index: int, separator: str = ": ") -> str:
        """
        Trasforma una riga in testo "feature: valore" per l'uso in un sistema RAG/LLM.
        I valori float sono arrotondati a 6 decimali.
        Questo metodo è privato (con underscore).
        """
        if row_index < 0 or row_index >= len(self.feat_data):
            raise ValueError(f"Indice {row_index} non valido.")
        row = self.feat_data.iloc[row_index]
        parts = []
        for feat in self.feature_names:
            value = row[feat]
            value_str = f"{value:.6f}" if isinstance(value, float) else str(value)
            parts.append(f"{feat}{separator}{value_str}")
        return ", ".join(parts)

    def save(self, path: str):
        """
        Serializza il dataset in formato testuale (lista di stringhe e target).
        Il file salvato conterrà una tupla (list_of_texts, target_series).
        """
        texts = [self._row_to_text(i) for i in range(len(self))]
        with open(path, 'wb') as f:
            pickle.dump((texts, self.target_data), f)
        logger.info(f"Dataset salvato in formato testuale in {path}")

    def text_to_dataset(self) -> 'TextDataset':
        """
        Converte il dataset corrente in un oggetto TextDataset,
        senza passare attraverso la serializzazione su disco.
        Restituisce un'istanza di TextDataset.
        """
        from text_dataset import TextDataset
        texts = [self._row_to_text(i) for i in range(len(self))]
        return TextDataset.from_data(texts, self.target_data)

    def __len__(self) -> int:
        return len(self.feat_data) if self.feat_data is not None else 0