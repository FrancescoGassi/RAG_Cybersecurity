from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BALANCE_TEST,
    BALANCE_TRAINING,
    CACHE_DIR,
    CACHE_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_FEATURES_PER_CHUNK,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SAMPLE_BATCH_SIZE,
    EMBEDDING_TOP_FEATURES_PER_SAMPLE,
    FORCE_REBUILD_INDEX,
    HYBRID_MAJORITY_WEIGHT,
    K_NEIGHBORS,
    OUTPUT_CSV,
    RANDOM_SEED,
    SHOW_PROGRESS,
    TARGET_COLUMN,
    TEST_CSV,
    TEST_LIMIT,
    TOP_K_FEATURES,
    TRAIN_CSV,
    TRAIN_SAMPLE_SIZE,
)
from src.dataset import Dataset
from src.faiss_rag_index import FaissRAGIndex


def label_name(label: int) -> str:
    return "malware" if int(label) == 1 else "goodware"


def safe_model_name(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return result or "embedding_model"


def file_signature(path: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset non trovato: {path}")
    stat = file_path.stat()
    return f"{file_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def cache_signature(feature_names: list[str]) -> str:
    values = [
        file_signature(TRAIN_CSV),
        str(TRAIN_SAMPLE_SIZE),
        str(BALANCE_TRAINING),
        str(RANDOM_SEED),
        str(TOP_K_FEATURES),
        EMBEDDING_MODEL_NAME,
        str(EMBEDDING_TOP_FEATURES_PER_SAMPLE),
        str(EMBEDDING_FEATURES_PER_CHUNK),
        *feature_names,
    ]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def validate_config() -> None:
    if K_NEIGHBORS <= 0:
        raise ValueError("K_NEIGHBORS deve essere maggiore di zero.")
    if not 0.0 < HYBRID_MAJORITY_WEIGHT < 1.0:
        raise ValueError(
            "HYBRID_MAJORITY_WEIGHT deve essere strettamente tra 0 e 1 "
            "per ottenere un voto realmente ibrido."
        )
    if EMBEDDING_BATCH_SIZE <= 0 or EMBEDDING_SAMPLE_BATCH_SIZE <= 0:
        raise ValueError("I batch size devono essere maggiori di zero.")
    file_signature(TRAIN_CSV)
    file_signature(TEST_CSV)


def load_data() -> tuple[Dataset, Dataset, list[str]]:
    print("\n1. Caricamento dataset...")
    train = Dataset.from_csv(
        TRAIN_CSV,
        TARGET_COLUMN,
        TRAIN_SAMPLE_SIZE,
        RANDOM_SEED,
        BALANCE_TRAINING,
    )
    test = Dataset.from_csv(
        TEST_CSV,
        TARGET_COLUMN,
        TEST_LIMIT,
        RANDOM_SEED,
        BALANCE_TEST,
    )

    if train.X.columns.tolist() != test.X.columns.tolist():
        raise ValueError("Training e test devono avere le stesse feature nello stesso ordine.")

    print(f"   Training: {len(train)}, classi={train.class_counts()}")
    print(f"   Test:     {len(test)}, classi={test.class_counts()}")

    print("\n2. Selezione feature con Mutual Information...")
    selected_train, feature_names, scores = train.select_best_features(
        TOP_K_FEATURES,
        RANDOM_SEED,
    )
    selected_test = test.select_features(feature_names)

    if TOP_K_FEATURES is not None and scores:
        for position, feature in enumerate(feature_names[:10], start=1):
            print(f"   {position:2d}. {feature}: {scores[feature]:.6f}")
    else:
        print(f"   Usate tutte le {len(feature_names)} feature (nessuna selezione MI).")

    return selected_train, selected_test, feature_names


def create_index() -> FaissRAGIndex:
    return FaissRAGIndex(
        model_name=EMBEDDING_MODEL_NAME,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        sample_batch_size=EMBEDDING_SAMPLE_BATCH_SIZE,
        top_features_per_sample=EMBEDDING_TOP_FEATURES_PER_SAMPLE,
        features_per_chunk=EMBEDDING_FEATURES_PER_CHUNK,
        device=EMBEDDING_DEVICE,
        show_progress=SHOW_PROGRESS,
    )


def get_index(train: Dataset, feature_names: list[str]) -> FaissRAGIndex:
    signature = cache_signature(feature_names)
    prefix = Path(CACHE_DIR) / safe_model_name(EMBEDDING_MODEL_NAME) / CACHE_NAME

    if not FORCE_REBUILD_INDEX and FaissRAGIndex.exists(prefix):
        index = create_index()
        index.load(prefix)
        if index.cache_signature == signature:
            print(f"\n3. Cache caricata: {prefix}_metadata.pkl")
            return index
        print("\n3. Cache non compatibile: ricostruzione.")

    print("\n3. Creazione embedding e indice FAISS...")
    index = create_index()
    index.fit(train.X, train.y.to_numpy(dtype=np.int64), signature)
    index.save(prefix)
    print(f"   Creato PKL: {prefix}_metadata.pkl")
    print(f"   Creato FAISS: {prefix}.faiss")
    return index


def main() -> None:
    # ========== BANNER ==========
    print("=" * 80)
    print("                      RAG con MAJORITY VOTING (senza LLM)                       ")
    print("=" * 80)
    print(f"Modello embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Vicini FAISS (k):  {K_NEIGHBORS}")
    if TOP_K_FEATURES is None:
        print(f"Selezione feature: USATE TUTTE")
    else:
        print(f"Selezione feature: TOP {TOP_K_FEATURES} (via MI)")
    print("=" * 80)
    # ====================================

    validate_config()
    train, test, feature_names = load_data()
    if K_NEIGHBORS > len(train):
        raise ValueError("K_NEIGHBORS supera il numero di campioni di training.")

    index = get_index(train, feature_names)

    print("\n4. Retrieval e Hybrid Majority Voting...")
    predictions = index.predict_many(
        test.X,
        K_NEIGHBORS,
        HYBRID_MAJORITY_WEIGHT,
    )
    y_true = test.y.to_numpy(dtype=np.int64)

    output = pd.DataFrame({
        "prediction": [label_name(value) for value in predictions],
        "true_label": [label_name(value) for value in y_true],
        "pred_type": ["HMV"] * len(predictions),
    })
    output.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision: {precision_score(y_true, predictions, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, predictions, average='macro', zero_division=0):.4f}")
    print(f"F1 macro:  {f1_score(y_true, predictions, average='macro', zero_division=0):.4f}")
    print("\nMatrice di confusione:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["goodware", "malware"],
        zero_division=0,
    ))
    print(f"CSV salvato in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()