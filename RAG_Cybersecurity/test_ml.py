"""
Test completo per Dataset: verifica caricamento, MI, ordinamento, split e serializzazione.
"""

import os
import pandas as pd
from src.dataset import Dataset
from src.text_dataset import TextDataset

def test_su_file(file_path, nome_file):
    print(f"--- Test su {nome_file} ---")
    sample_file = "temp_sample.csv"

    # Preleva un campione casuale di 1000 righe
    df = pd.read_csv(file_path).sample(n=min(1000, sum(1 for _ in open(file_path))), random_state=42)
    df.to_csv(sample_file, index=False)

    # Costruttore con solo file_path
    dataset = Dataset(sample_file)
    print(f"Caricati {len(dataset)} esempi, {len(dataset.feature_names)} feature")
    print(f"Target name rilevato: {dataset.target_name}")

    # Calcola MI
    mi = dataset.compute_mutual_information()
    print("Top 5 feature per MI:")
    for i, (feat, score) in enumerate(list(mi.items())[:5]):
        print(f"  {i+1}. {feat}: {score:.4f}")

    # Ordina per MI
    dataset_sorted = dataset.sort_features_by_mi(mi)
    print("Prime 5 feature dopo ordinamento:", dataset_sorted.feature_names[:5])

    # Split stratificato
    train, test = dataset_sorted.train_test_split(test_size=0.2, random_state=42)
    print(f"Split: train={len(train)}, test={len(test)}")

    # Distribuzione classi
    train_dist = train.target_data.value_counts(normalize=True)
    test_dist = test.target_data.value_counts(normalize=True)
    print(f"Distribuzione train: {dict(train_dist)}")
    print(f"Distribuzione test:  {dict(test_dist)}")

    # Verifica la serializzazione e la conversione in testo
    train.save('temp_train.pkl')
    test.save('temp_test.pkl')

    # Carica come TextDataset per verificare la conversione
    train_text = TextDataset('temp_train.pkl')
    test_text = TextDataset('temp_test.pkl')

    # Mostra un esempio di testo
    print("Esempio di testo (prima riga del training set):")
    print(train_text.texts[0][:200])  # primi 200 caratteri

    # Rimuovi file temporanei
    os.remove(sample_file)
    os.remove('temp_train.pkl')
    os.remove('temp_test.pkl')
    print("✅ Test completato con successo!\n")

def main():
    print("=== Test Dataset ===\n")
    test_su_file('DatasetPE/BODMAS_features_named.csv', 'training (BODMAS_features_named)')
    test_su_file('DatasetPE/test_named.csv', 'test (test_named)')

if __name__ == "__main__":
    main()