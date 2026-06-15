from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.config import (
    BALANCE_TEST,
    BALANCE_TRAINING,
    K_NEIGHBORS,
    MAJORITY_VOTING_OUTPUT_CSV,
    RANDOM_SEED,
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


def main() -> None:
    if K_NEIGHBORS <= 0 or K_NEIGHBORS % 2 == 0:
        raise ValueError(
            "K_NEIGHBORS deve essere dispari e maggiore di zero."
        )

    print("=" * 80)
    print(" RAG con MAJORITY VOTING (senza LLM) ".center(80))
    print("=" * 80)
    print(f"Training limit:     {TRAIN_SAMPLE_SIZE}")
    print(f"Test limit:         {TEST_LIMIT}")
    print(f"Top MI features:    {TOP_K_FEATURES}")
    print(f"Vicini FAISS (k):   {K_NEIGHBORS}")
    print(f"Output CSV:         {MAJORITY_VOTING_OUTPUT_CSV}")
    print("=" * 80)

    print("\n1. Caricamento dei dataset...")

    train = Dataset.from_csv(
        TRAIN_CSV,
        limit=TRAIN_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
        balanced=BALANCE_TRAINING,
    )
    test = Dataset.from_csv(
        TEST_CSV,
        limit=TEST_LIMIT,
        random_state=RANDOM_SEED,
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

    print("\n2. Selezione feature tramite Mutual Information...")

    mi_scores = train.compute_mutual_information(
        random_state=RANDOM_SEED
    )
    features = list(mi_scores)[:TOP_K_FEATURES]

    for rank, feature in enumerate(features[:10], start=1):
        print(f"   {rank:2d}. {feature}: {mi_scores[feature]:.6f}")

    train = train.select_features(features)
    test = test.select_features(features)

    print("\n3. Costruzione indice FAISS numerico...")

    index = FaissRAGIndex()
    index.fit(
        train.feat_data,
        train.target_data.to_numpy(dtype=int),
        features,
    )

    print("\n4. Predizione mediante Majority Voting...")

    csv_rows: list[dict[str, str]] = []
    full_rows: list[dict[str, object]] = []

    for sample_index, row in enumerate(
        test.feat_data.itertuples(index=False, name=None)
    ):
        prediction, counts, _ = index.majority_vote(row, K_NEIGHBORS)
        true_label = int(test.target_data.iloc[sample_index])

        csv_rows.append(
            {
                "prediction": label_name(prediction),
                "true_label": label_name(true_label),
                "pred_type": "MV",
            }
        )

        full_rows.append(
            {
                "prediction_num": int(prediction),
                "true_label_num": true_label,
                "prediction": label_name(prediction),
                "true_label": label_name(true_label),
                "pred_type": "MV",
                "decision_source": "majority_voting",
                "mv_goodware_votes": int(counts[0]),
                "mv_malware_votes": int(counts[1]),
            }
        )

    if not csv_rows:
        raise RuntimeError("Nessuna predizione prodotta.")

    pd.DataFrame(
        csv_rows,
        columns=["prediction", "true_label", "pred_type"],
    ).to_csv(MAJORITY_VOTING_OUTPUT_CSV, index=False)

    result = pd.DataFrame(full_rows)
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
    print(f"\nCSV salvato in: {MAJORITY_VOTING_OUTPUT_CSV}")


if __name__ == "__main__":
    main()