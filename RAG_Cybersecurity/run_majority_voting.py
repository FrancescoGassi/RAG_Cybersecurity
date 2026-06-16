from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.config import (
    BALANCE_TEST,
    BALANCE_TRAINING,
    CACHE_DIR,
    CACHE_VERSION,
    CHECK_TRAIN_TEST_OVERLAP,
    EMBEDDING_ENFORCE_MAX_LENGTH,
    EMBEDDING_FEATURES_PER_CHUNK,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_SORT_FEATURES_BY_MAGNITUDE,
    EMBEDDING_TOP_FEATURES_PER_SAMPLE,
    EMBEDDING_VALUE_DECIMALS,
    FAIL_ON_TRAIN_TEST_OVERLAP,
    FORCE_REBUILD_INDEX,
    K_NEIGHBORS,
    MAJORITY_VOTING_METRICS_JSON,
    MAJORITY_VOTING_OUTPUT_CSV,
    MUTUAL_INFORMATION_N_NEIGHBORS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_CSV,
    TEST_LIMIT,
    TOP_K_FEATURES,
    TRAIN_CSV,
    TRAIN_SAMPLE_SIZE,
    VALIDATE_BODMAS_FEATURE_NAMES,
    VOTING_STRATEGY,
)
from src.dataset import Dataset
from src.faiss_rag_index import FaissRAGIndex
from src.feature_names import BODMASFeatureNames
from src.feature_text_encoder import TEXT_ENCODING_VERSION


def label_name(label: int) -> str:
    label = int(label)
    if label == 0:
        return "goodware"
    if label == 1:
        return "malware"
    raise ValueError(f"Etichetta non binaria: {label}")


def safe_model_name(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    if not safe:
        raise ValueError("Nome del modello non valido per il percorso cache.")
    return safe


def file_signature(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"missing:{file_path.resolve()}"
    stat = file_path.stat()
    return (
        f"{file_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    )


def build_cache_signature(
    model_name: str,
    feature_names: list[str],
) -> str:
    fields = [
        CACHE_VERSION,
        TEXT_ENCODING_VERSION,
        file_signature(TRAIN_CSV),
        str(TARGET_COLUMN),
        str(TRAIN_SAMPLE_SIZE),
        str(BALANCE_TRAINING),
        str(RANDOM_SEED),
        str(MUTUAL_INFORMATION_N_NEIGHBORS),
        model_name,
        str(EMBEDDING_TOP_FEATURES_PER_SAMPLE),
        str(EMBEDDING_FEATURES_PER_CHUNK),
        str(EMBEDDING_VALUE_DECIMALS),
        str(EMBEDDING_SORT_FEATURES_BY_MAGNITUDE),
        str(EMBEDDING_ENFORCE_MAX_LENGTH),
        *feature_names,
    ]
    return hashlib.sha256(
        "\n".join(fields).encode("utf-8")
    ).hexdigest()


def validate_config() -> None:
    if TOP_K_FEATURES <= 0:
        raise ValueError("TOP_K_FEATURES deve essere maggiore di zero.")
    if MUTUAL_INFORMATION_N_NEIGHBORS <= 0:
        raise ValueError(
            "MUTUAL_INFORMATION_N_NEIGHBORS deve essere maggiore di zero."
        )
    if K_NEIGHBORS <= 0:
        raise ValueError("K_NEIGHBORS deve essere maggiore di zero.")
    if VOTING_STRATEGY not in {"majority", "similarity_weighted"}:
        raise ValueError(
            "VOTING_STRATEGY deve essere 'majority' oppure "
            "'similarity_weighted'."
        )
    if VOTING_STRATEGY == "majority" and K_NEIGHBORS % 2 == 0:
        raise ValueError(
            "Con Majority Voting, K_NEIGHBORS deve essere dispari."
        )
    if EMBEDDING_TOP_FEATURES_PER_SAMPLE <= 0:
        raise ValueError(
            "EMBEDDING_TOP_FEATURES_PER_SAMPLE deve essere positivo."
        )
    if EMBEDDING_FEATURES_PER_CHUNK <= 0:
        raise ValueError(
            "EMBEDDING_FEATURES_PER_CHUNK deve essere positivo."
        )
    if EMBEDDING_VALUE_DECIMALS < 0:
        raise ValueError(
            "EMBEDDING_VALUE_DECIMALS non può essere negativo."
        )

    for path in (TRAIN_CSV, TEST_CSV):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dataset non trovato: {path}")


def validate_feature_schema(train: Dataset, test: Dataset) -> None:
    if train.feature_names != test.feature_names:
        train_set = set(train.feature_names)
        test_set = set(test.feature_names)
        missing_in_test = sorted(train_set - test_set)
        extra_in_test = sorted(test_set - train_set)
        same_set = train_set == test_set
        if same_set:
            raise ValueError(
                "Training e test contengono le stesse feature, ma in ordine "
                "diverso. L'ordine deve coincidere esattamente."
            )
        raise ValueError(
            "Training e test non hanno le stesse feature. "
            f"Mancanti nel test: {missing_in_test[:10]}; "
            f"extra nel test: {extra_in_test[:10]}."
        )

    if VALIDATE_BODMAS_FEATURE_NAMES:
        expected = BODMASFeatureNames.get_all_feature_names()
        if train.feature_names != expected:
            first_mismatch = next(
                (
                    index
                    for index, (actual, wanted) in enumerate(
                        zip(train.feature_names, expected)
                    )
                    if actual != wanted
                ),
                None,
            )
            raise ValueError(
                "Le colonne non coincidono con lo schema BODMAS atteso "
                f"({len(expected)} feature). Feature presenti: "
                f"{len(train.feature_names)}; prima posizione diversa: "
                f"{first_mismatch}."
            )


def report_data_integrity(train: Dataset, test: Dataset) -> None:
    train_duplicates = train.duplicate_row_count()
    test_duplicates = test.duplicate_row_count()
    print(
        "   Duplicati interni: "
        f"training={train_duplicates}, test={test_duplicates}"
    )

    if not CHECK_TRAIN_TEST_OVERLAP:
        return

    overlap = train.overlap_summary(test)
    print(
        "   Sovrapposizione train/test: "
        f"righe uniche condivise="
        f"{overlap['shared_unique_feature_rows']}, "
        f"righe training coinvolte="
        f"{overlap['training_rows_in_overlap']}, "
        f"righe test coinvolte={overlap['test_rows_in_overlap']}"
    )
    if overlap["shared_unique_feature_rows"] > 0:
        message = (
            "Sono presenti vettori di feature identici in training e test. "
            "Le metriche possono risultare artificialmente elevate."
        )
        if FAIL_ON_TRAIN_TEST_OVERLAP:
            raise ValueError(message)
        print(f"   AVVISO: {message}")


def load_and_prepare_data() -> tuple[Dataset, Dataset, list[str]]:
    print("\n1. Caricamento dei dataset...")
    train = Dataset.from_csv(
        TRAIN_CSV,
        target_column=TARGET_COLUMN,
        limit=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        balanced=BALANCE_TRAINING,
    )
    test = Dataset.from_csv(
        TEST_CSV,
        target_column=TARGET_COLUMN,
        limit=TEST_LIMIT,
        random_state=RANDOM_SEED,
        balanced=BALANCE_TEST,
    )

    print(f"   Training: {len(train)}, classi={train.class_counts()}")
    print(f"   Test:     {len(test)}, classi={test.class_counts()}")
    validate_feature_schema(train, test)
    report_data_integrity(train, test)

    print("\n2. Selezione delle feature tramite Mutual Information...")
    mi_scores = train.compute_mutual_information(
        random_state=RANDOM_SEED,
        n_neighbors=MUTUAL_INFORMATION_N_NEIGHBORS,
    )
    feature_names = list(mi_scores)[: min(TOP_K_FEATURES, len(mi_scores))]
    if not feature_names:
        raise RuntimeError("La selezione non ha restituito feature.")

    for rank, feature in enumerate(feature_names[:10], start=1):
        print(f"   {rank:2d}. {feature}: {mi_scores[feature]:.6f}")

    effective_top = min(
        EMBEDDING_TOP_FEATURES_PER_SAMPLE,
        len(feature_names),
    )
    print(
        f"   Feature selezionate globalmente: {len(feature_names)}; "
        f"usate per campione: {effective_top}; "
        f"feature per chunk: {EMBEDDING_FEATURES_PER_CHUNK}"
    )

    return (
        train.select_features(feature_names),
        test.select_features(feature_names),
        feature_names,
    )


def new_index(model_name: str) -> FaissRAGIndex:
    return FaissRAGIndex(embedding_model_name=model_name)


def get_or_build_index(
    train: Dataset,
    feature_names: list[str],
    model_name: str,
) -> FaissRAGIndex:
    cache_signature = build_cache_signature(model_name, feature_names)
    cache_prefix = (
        Path(CACHE_DIR)
        / safe_model_name(model_name)
        / "training_index"
    )

    if not FORCE_REBUILD_INDEX and FaissRAGIndex.exists(cache_prefix):
        cached_index = new_index(model_name)
        try:
            cached_index.load(cache_prefix)
        except Exception as exc:
            print(
                "\n3. Cache non leggibile; ricostruzione necessaria: "
                f"{exc}"
            )
        else:
            if cached_index.cache_signature == cache_signature:
                print(f"\n3. Cache FAISS valida caricata da {cache_prefix}")
                return cached_index
            print("\n3. Cache non compatibile: ricostruzione necessaria.")
    else:
        print("\n3. Costruzione degli embedding e dell'indice FAISS...")

    index = new_index(model_name)
    index.fit(
        train.feat_data,
        train.target_data.to_numpy(dtype=np.int64),
        feature_names,
        cache_signature=cache_signature,
    )
    index.save(cache_prefix)
    print(f"   Cache salvata in: {cache_prefix}")
    return index


def evaluate_model(
    train: Dataset,
    test: Dataset,
    feature_names: list[str],
    model_name: str,
    output_csv: str,
    metrics_json: str,
    print_report: bool = True,
) -> dict[str, object]:
    if K_NEIGHBORS > len(train):
        raise ValueError(
            f"K_NEIGHBORS={K_NEIGHBORS} supera il numero di campioni "
            f"di training ({len(train)})."
        )

    index = get_or_build_index(train, feature_names, model_name)

    print("\n4. Embedding del test, retrieval FAISS e voto...")
    vote_results = index.vote_many(
        test.feat_data,
        K_NEIGHBORS,
        strategy=VOTING_STRATEGY,
    )
    if len(vote_results) != len(test):
        raise RuntimeError(
            "Il numero di predizioni non coincide con il test set."
        )

    rows: list[dict[str, object]] = []
    for sample_index, vote in enumerate(vote_results):
        true_label = int(test.target_data.iloc[sample_index])
        neighbors = vote.neighbors
        rows.append(
            {
                "sample_index": sample_index,
                "source_row_index": int(test.source_indices[sample_index]),
                "prediction": label_name(vote.prediction),
                "true_label": label_name(true_label),
                "prediction_num": int(vote.prediction),
                "true_label_num": true_label,
                "pred_type": "MV" if VOTING_STRATEGY == "majority" else "WMV",
                "decision_source": (
                    "embedding_faiss_" + VOTING_STRATEGY
                ),
                "embedding_model": model_name,
                "k_neighbors_requested": K_NEIGHBORS,
                "k_neighbors_effective": len(neighbors),
                "goodware_votes": int(vote.counts[0]),
                "malware_votes": int(vote.counts[1]),
                "goodware_similarity_score": float(
                    vote.similarity_scores[0]
                ),
                "malware_similarity_score": float(
                    vote.similarity_scores[1]
                ),
                "nearest_similarity": float(neighbors[0].similarity),
                "neighbor_indices": ";".join(
                    str(neighbor.index) for neighbor in neighbors
                ),
                "neighbor_labels": ";".join(
                    label_name(neighbor.label) for neighbor in neighbors
                ),
                "neighbor_similarities": ";".join(
                    f"{neighbor.similarity:.8f}"
                    for neighbor in neighbors
                ),
            }
        )

    result = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    y_true = result["true_label_num"].to_numpy(dtype=np.int64)
    y_pred = result["prediction_num"].to_numpy(dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    summary: dict[str, object] = {
        "embedding_model": model_name,
        "voting_strategy": VOTING_STRATEGY,
        "n_training": len(train),
        "n_test": len(test),
        "selected_features": len(feature_names),
        "features_per_sample": min(
            EMBEDDING_TOP_FEATURES_PER_SAMPLE,
            len(feature_names),
        ),
        "features_per_chunk": EMBEDDING_FEATURES_PER_CHUNK,
        "k_neighbors": K_NEIGHBORS,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_malware": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_malware": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_malware": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "confusion_matrix": matrix.tolist(),
        "output_csv": str(output_path),
    }

    metrics_path = Path(metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    if print_report:
        print("\n" + "=" * 80)
        print(f"MODELLO EMBEDDING: {model_name}")
        print(f"STRATEGIA DI VOTO: {VOTING_STRATEGY}")
        print(f"ACCURATEZZA:       {summary['accuracy'] * 100:.2f}%")
        print(f"BALANCED ACCURACY: {summary['balanced_accuracy']:.4f}")
        print(f"PRECISION MACRO:   {summary['precision_macro']:.4f}")
        print(f"RECALL MACRO:      {summary['recall_macro']:.4f}")
        print(f"F1 MACRO:          {summary['f1_macro']:.4f}")
        print(f"MCC:               {summary['matthews_correlation_coefficient']:.4f}")

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
        print(f"Predizioni salvate in: {output_path}")
        print(f"Metriche salvate in:   {metrics_path}")

    return summary


def main() -> None:
    validate_config()

    print("=" * 80)
    print(" RETRIEVAL FAISS CON MAJORITY VOTING (SENZA LLM) ".center(80))
    print("=" * 80)
    print(f"Modello embedding: {EMBEDDING_MODEL_NAME}")
    print(f"Vicini FAISS (k):  {K_NEIGHBORS}")
    print(f"Strategia di voto: {VOTING_STRATEGY}")
    print("=" * 80)

    train, test, feature_names = load_and_prepare_data()
    evaluate_model(
        train=train,
        test=test,
        feature_names=feature_names,
        model_name=EMBEDDING_MODEL_NAME,
        output_csv=MAJORITY_VOTING_OUTPUT_CSV,
        metrics_json=MAJORITY_VOTING_METRICS_JSON,
        print_report=True,
    )


if __name__ == "__main__":
    main()
