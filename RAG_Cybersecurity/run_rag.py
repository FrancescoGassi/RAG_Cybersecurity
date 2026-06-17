from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.config import (
    BALANCE_TEST,
    BALANCE_TRAINING,
    CACHE_DIR,
    CACHE_VERSION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_FEATURES_PER_CHUNK,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SAMPLE_BATCH_SIZE,
    EMBEDDING_TOP_FEATURES_PER_SAMPLE,
    FORCE_REBUILD_INDEX,
    K_NEIGHBORS,
    LLM_MODEL_NAME,
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
from src.llm_predictor import LLMPredictor


def file_signature(path: str) -> dict[str, object]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_signature(features: list[str], train: Dataset) -> str:
    payload = {
        "version": CACHE_VERSION,
        "train_file": file_signature(TRAIN_CSV),
        "training_rows": len(train),
        "training_classes": train.class_counts(),
        "training_limit": TRAIN_SAMPLE_SIZE,
        "balance_training": BALANCE_TRAINING,
        "features": features,
        "seed": RANDOM_SEED,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "top_features_per_sample": EMBEDDING_TOP_FEATURES_PER_SAMPLE,
        "features_per_chunk": EMBEDDING_FEATURES_PER_CHUNK,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def print_header() -> None:
    print("=" * 80)
    print(" RAG con LLM ".center(80))
    print("=" * 80)
    print(f"LLM:                 {LLM_MODEL_NAME}")
    print(f"Training limit:      {TRAIN_SAMPLE_SIZE}")
    print(f"Test limit:          {TEST_LIMIT}")
    print(f"Top MI features:     {TOP_K_FEATURES}")
    print(f"Vicini FAISS (k):    {K_NEIGHBORS}")
    print("Fallback:            Majority Voting")
    print(f"Output CSV:          {OUTPUT_CSV}")
    print("=" * 80)


def main() -> None:
    if K_NEIGHBORS <= 0 or K_NEIGHBORS % 2 == 0:
        raise ValueError(
            "K_NEIGHBORS deve essere dispari e maggiore di zero."
        )

    print_header()

    print("\n1. Caricamento dei dataset...")

    train = Dataset.from_csv(
        TRAIN_CSV,
        target_column=TARGET_COLUMN,
        limit=TRAIN_SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
        balanced=BALANCE_TRAINING,
    )
    test = Dataset.from_csv(
        TEST_CSV,
        target_column=TARGET_COLUMN,
        limit=TEST_LIMIT,
        random_seed=RANDOM_SEED,
        balanced=BALANCE_TEST,
    )

    print(
        f"   Training usato: {len(train)}, "
        f"classi={train.class_counts()}"
    )
    print(
        f"   Test usato:     {len(test)}, "
        f"classi={test.class_counts()}"
    )

    print("\n2. Selezione feature con Mutual Information...")

    train_selected, selected_features, mi_scores = train.select_best_features(
        TOP_K_FEATURES,
        RANDOM_SEED,
    )
    test_selected = test.select_features(selected_features)

    for rank, feature in enumerate(selected_features[:10], start=1):
        print(f"   {rank:2d}. {feature}: {mi_scores[feature]:.6f}")

    print("\n3. Costruzione/caricamento indice FAISS numerico...")

    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Usiamo un prefisso specifico per RAG
    index_prefix = cache_dir / "bodmas_rag_llm"

    expected_signature = cache_signature(selected_features, train_selected)

    # Crea l'indice con i parametri di embedding
    index = FaissRAGIndex(
        model_name=EMBEDDING_MODEL_NAME,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        sample_batch_size=EMBEDDING_SAMPLE_BATCH_SIZE,
        top_features_per_sample=EMBEDDING_TOP_FEATURES_PER_SAMPLE,
        features_per_chunk=EMBEDDING_FEATURES_PER_CHUNK,
        device=EMBEDDING_DEVICE,
        show_progress=SHOW_PROGRESS,
    )

    cache_valid = False
    if (
        not FORCE_REBUILD_INDEX
        and FaissRAGIndex.exists(index_prefix)
    ):
        index.load(index_prefix)
        cache_valid = index.cache_signature == expected_signature
        if not cache_valid:
            print(
                "   Cache obsoleta o incompatibile: verrà ricostruita."
            )

    if FORCE_REBUILD_INDEX or not cache_valid:
        index.fit(
            train_selected.X,
            train_selected.y.to_numpy(dtype=np.int64),
            cache_signature=expected_signature,
        )
        index.save(index_prefix)
        print("   Indice FAISS creato e salvato.")
    else:
        print("   Indice FAISS valido caricato dalla cache.")

    print("\n4. Caricamento LLM...")

    predictor = LLMPredictor(
        model_name=LLM_MODEL_NAME,
        debug=False,
        use_4bit=True,
    )
    predictor.load()

    print("\n5. Predizione RAG...")

    result = predictor.predict(
        test_selected.X,
        test_selected.y.astype(int).tolist(),
        index,
        OUTPUT_CSV,
        k=K_NEIGHBORS,
    )

    if result.empty:
        raise RuntimeError("Nessuna predizione prodotta.")

    y_true = result["true_label_num"].astype(int)
    y_pred = result["prediction_num"].astype(int)

    accuracy = float((y_true == y_pred).mean())

    print("\n" + "=" * 80)
    print(f"ACCURATEZZA: {accuracy * 100:.2f}%")

    print("\nDistribuzione predizioni finali:")
    print(result["prediction"].value_counts().to_string())

    print("\nOrigine decisioni:")
    print(result["pred_type"].value_counts().to_string())

    print("\nDettaglio sorgenti interne:")
    print(result["decision_source"].value_counts().to_string())

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nMatrice di confusione:")
    print(
        pd.DataFrame(
            matrix,
            index=["true_goodware", "true_malware"],
            columns=["pred_goodware", "pred_malware"],
        ).to_string()
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["goodware", "malware"],
            zero_division=0,
        )
    )

    print("=" * 80)
    print(f"\nCSV salvato in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()