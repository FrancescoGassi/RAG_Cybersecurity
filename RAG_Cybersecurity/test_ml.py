import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.dataset import Dataset
from src.text_dataset import TextDataset

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def test_su_file(file_path, nome_file):
    print(f"--- Test su {nome_file} ---")
    sample_file = os.path.join(CACHE_DIR, "temp_sample.csv")

    df = pd.read_csv(file_path).sample(n=min(1000, sum(1 for _ in open(file_path))), random_state=42)
    df.to_csv(sample_file, index=False)

    dataset = Dataset(sample_file)
    print(f"Caricati {len(dataset)} esempi, {len(dataset.feature_names)} feature")
    print(f"Target name rilevato: {dataset.target_name}")

    mi = dataset.compute_mutual_information()
    print("Top 5 feature per MI:")
    for i, (feat, score) in enumerate(list(mi.items())[:5]):
        print(f"  {i+1}. {feat}: {score:.4f}")

    dataset_sorted = dataset.sort_features_by_mi(mi)
    print("Prime 5 feature dopo ordinamento:", dataset_sorted.feature_names[:5])

    train, test = dataset_sorted.train_test_split(test_size=0.2, random_state=42)
    print(f"Split: train={len(train)}, test={len(test)}")

    train_dist = train.target_data.value_counts(normalize=True)
    test_dist = test.target_data.value_counts(normalize=True)
    print(f"Distribuzione train: { {k: float(v) for k, v in train_dist.items()} }")
    print(f"Distribuzione test:  { {k: float(v) for k, v in test_dist.items()} }")

    train_pkl = os.path.join(CACHE_DIR, 'temp_train.pkl')
    test_pkl = os.path.join(CACHE_DIR, 'temp_test.pkl')
    train.save(train_pkl)
    test.save(test_pkl)

    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)

    print("Esempio di testo (prima riga del training set):")
    print(train_text.get_texts()[0][:200])

    os.remove(sample_file)
    os.remove(train_pkl)
    os.remove(test_pkl)
    print("✅ Test completato con successo!\n")

def main():
    print("=== Test Dataset ===\n")
    test_su_file('DatasetPE/BODMAS_features_named.csv', 'training (BODMAS_features_named)')
    test_su_file('DatasetPE/test_named.csv', 'test (test_named)')

if __name__ == "__main__":
    main()