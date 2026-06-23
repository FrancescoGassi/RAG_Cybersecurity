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
    K_NEIGHBORS,
    OUTPUT_CSV,
    RANDOM_SEED,
    RETRIEVAL_MODE,
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
        RETRIEVAL_MODE,
        *feature_names,
    ]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def mutual_info_cache_signature() -> str:
    values = [
        file_signature(TRAIN_CSV),
        str(TRAIN_SAMPLE_SIZE),
        str(BALANCE_TRAINING),
        str(RANDOM_SEED),
        str(TARGET_COLUMN),
    ]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def get_mutual_info_scores(train: Dataset) -> dict[str, float]:
    sig = mutual_info_cache_signature()
    cache_dir = Path(CACHE_DIR) / "mutual_info"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{sig}.csv"

    if cache_file.exists():
        print("   Caricamento punteggi MI da cache...")
        df = pd.read_csv(cache_file)
        return {row['feature']: row['score'] for _, row in df.iterrows()}

    print("   Calcolo Mutual Information su tutte le feature (prima esecuzione o dati cambiati)...")
    scores = train.compute_mutual_info(RANDOM_SEED)
    df = pd.DataFrame(list(scores.items()), columns=['feature', 'score'])
    df.to_csv(cache_file, index=False)
    print(f"   Punteggi MI salvati in {cache_file}")
    return scores


def validate_config() -> None:
    if K_NEIGHBORS <= 0:
        raise ValueError("K_NEIGHBORS deve essere maggiore di zero.")
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

    print("\n2. Selezione feature con Mutual Information (cache abilitata)...")
    mi_scores = get_mutual_info_scores(train)
    sorted_features = sorted(mi_scores.items(), key=lambda x: -x[1])

    if TOP_K_FEATURES is not None and TOP_K_FEATURES < len(sorted_features):
        selected_features = [f for f, _ in sorted_features[:TOP_K_FEATURES]]
    else:
        selected_features = [f for f, _ in sorted_features]

    for pos, (feature, score) in enumerate(sorted_features[:10], start=1):
        print(f"   {pos:2d}. {feature}: {score:.6f}")
    if TOP_K_FEATURES is not None:
        print(f"   Selezionate {len(selected_features)} feature su {len(sorted_features)} totali.")
    else:
        print(f"   Usate tutte le {len(selected_features)} feature (nessuna selezione).")

    selected_train = train.select_features(selected_features)
    selected_test = test.select_features(selected_features)
    return selected_train, selected_test, selected_features


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
    print("=" * 80)
    print("                    RAG con PURE MAJORITY VOTING (solo conteggio)                     ")
    print("=" * 80)
    print(f"Modello embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Vicini FAISS (k):  {K_NEIGHBORS}")
    if TOP_K_FEATURES is None:
        print(f"Selezione feature: USATE TUTTE")
    else:
        print(f"Selezione feature: TOP {TOP_K_FEATURES} (via MI)")
    print(f"Modalità retrieval: {RETRIEVAL_MODE.upper()}")
    print("=" * 80)

    validate_config()
    train, test, feature_names = load_data()
    if K_NEIGHBORS > len(train):
        raise ValueError("K_NEIGHBORS supera il numero di campioni di training.")

    index = get_index(train, feature_names)

    pred_type_label = "PMV"

    print(f"\n4. Retrieval e Majority Voting (puro)...")
    predictions = index.predict_many(test.X, K_NEIGHBORS)
    y_true = test.y.to_numpy(dtype=np.int64)

    base, ext = OUTPUT_CSV.rsplit(".", 1)
    output_csv = f"{base}_{RETRIEVAL_MODE}.{ext}"

    output = pd.DataFrame({
        "prediction": [label_name(v) for v in predictions],
        "true_label": [label_name(v) for v in y_true],
        "pred_type": [pred_type_label] * len(predictions),
        "retrieval_mode": [RETRIEVAL_MODE] * len(predictions),
    })
    output.to_csv(output_csv, index=False)

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
    print(f"CSV salvato in: {output_csv}")


if __name__ == "__main__":
    main()