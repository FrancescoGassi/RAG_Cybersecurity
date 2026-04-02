"""
Test completo del flusso RAG: da dataset originale a embedding.
"""

import sys
import os
import logging
import pandas as pd

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("dataset").setLevel(logging.INFO)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dataset import Dataset
from text_dataset import TextDataset
from embedding import Embedding

def main():
    print("=== Test flusso RAG ===\n")

    print("1. Caricamento campione del dataset...")
    sample_size = 1000
    df = pd.read_csv('DatasetPE/BODMAS_features_named.csv')
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    sample_file = 'temp_sample.csv'
    df_sample.to_csv(sample_file, index=False)

    ds = Dataset(sample_file)
    print(f"   Caricati {len(ds)} esempi, {len(ds.feature_names)} feature")

    print("2. Calcolo mutual information e ordinamento...")
    mi = ds.compute_mutual_information()
    sorted_ds = ds.sort_features_by_mi(mi)
    print("   Top 5 feature:", sorted_ds.feature_names[:5])

    print("3. Split in train/test...")
    train_ds, test_ds = sorted_ds.train_test_split(test_size=0.2, random_state=42)
    print(f"   Train: {len(train_ds)} esempi, Test: {len(test_ds)} esempi")

    print("4. Serializzazione in formato testuale...")
    train_ds.save('train_texts.pkl')
    test_ds.save('test_texts.pkl')
    print("   File salvati: train_texts.pkl, test_texts.pkl")

    print("5. Caricamento TextDataset...")
    train_text = TextDataset('train_texts.pkl')
    test_text = TextDataset('test_texts.pkl')
    print(f"   Train: {len(train_text.texts)} testi, Test: {len(test_text.texts)} testi")

    print("6. Creazione modello di embedding...")
    emb = Embedding()
    print("   Modello caricato (all-MiniLM-L6-v2)")

    print("7. Generazione embedding per il training set...")
    train_embeddings = train_text.text_to_emb(emb)
    print(f"   Shape embedding: {train_embeddings.embeddings.shape}")

    print("8. Generazione embedding per il test set...")
    test_embeddings = test_text.text_to_emb(emb)
    print(f"   Shape embedding: {test_embeddings.embeddings.shape}")

    print("9. Distribuzione classi:")
    print("   Train:", train_text.targets.value_counts().to_dict())
    print("   Test:", test_text.targets.value_counts().to_dict())

    os.remove(sample_file)
    print("\n✅ Flusso RAG completato con successo!")

if __name__ == "__main__":
    main()