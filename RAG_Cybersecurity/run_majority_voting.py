from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

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
        raise ValueError("K_NEIGHBORS deve essere dispari e maggiore di zero.")

    print("=" * 80)
    print(" FAISS + MAJORITY VOTING PURO (SENZA LLM) ".center(80))
    print("=" * 80)
    print(f"Training limit:     {TRAIN_SAMPLE_SIZE}")
    print(f"Test limit:         {TEST_LIMIT}")
    print(f"Top MI features:    {TOP_K_FEATURES}")
    print(f"Vicini FAISS (k):   {K_NEIGHBORS}")
    print(f"Output CSV:         {MAJORITY_VOTING_OUTPUT_CSV}")
    print("=" * 80)

    print("\n1. Caricamento dataset...")
    full_train = Dataset.from_csv(TRAIN_CSV)
    full_test = Dataset.from_csv(TEST_CSV)

    train = full_train.stratified_sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_SEED, balanced=BALANCE_TRAINING)
    test = full_test.stratified_sample(TEST_LIMIT, random_state=RANDOM_SEED, balanced=BALANCE_TEST)

    print(f"   Training usato: {len(train)}, classi={train.class_counts()}")
    print(f"   Test usato:     {len(test)}, classi={test.class_counts()}")

    print("\n2. Selezione feature tramite Mutual Information...")
    mi = train.compute_mutual_information(random_state=RANDOM_SEED)
    features = list(mi)[:TOP_K_FEATURES]
    for rank, feature in enumerate(features[:10], start=1):
        print(f"   {rank:2d}. {feature}: {mi[feature]:.6f}")

    train = train.select_features(features)
    test = test.select_features(features)

    print("\n3. Costruzione indice FAISS numerico...")
    index = FaissRAGIndex()
    index.fit(train.feat_data, train.target_data.to_numpy(dtype=int), features)

    print("\n4. Predizione mediante Majority Voting...")
    csv_rows: list[dict[str, str]] = []
    full_rows: list[dict[str, object]] = []

    for sample_index, row in enumerate(test.feat_data.itertuples(index=False, name=None)):
        prediction, counts, _neighbors = index.majority_vote(row, K_NEIGHBORS)
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
                "mv_goodware_votes": int(counts.get(0, 0)),
                "mv_malware_votes": int(counts.get(1, 0)),
            }
        )

    if not csv_rows:
        raise RuntimeError("Nessuna predizione prodotta.")

    pd.DataFrame(csv_rows, columns=["prediction", "true_label", "pred_type"]).to_csv(MAJORITY_VOTING_OUTPUT_CSV, index=False)
    result = pd.DataFrame(full_rows)

    y_true = result["true_label_num"].astype(int)
    y_pred = result["prediction_num"].astype(int)
    accuracy = float((y_true == y_pred).mean())

    print("\n" + "=" * 80)
    print(f"ACCURACY: {accuracy * 100:.2f}%")
    print("Distribuzione predizioni:", dict(Counter(result["prediction"].tolist())))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nMatrice di confusione:")
    print(pd.DataFrame(matrix, index=["true_goodware", "true_malware"], columns=["pred_goodware", "pred_malware"]).to_string())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["goodware", "malware"], zero_division=0))

    if result["prediction_num"].nunique() == 1:
        print("ATTENZIONE: il Majority Voting ha prodotto una sola classe.")

    print(f"CSV salvato in: {Path(MAJORITY_VOTING_OUTPUT_CSV).resolve()}")
    print("Colonne CSV: prediction,true_label,pred_type")
    print("=" * 80)

if __name__ == "__main__":
    main()