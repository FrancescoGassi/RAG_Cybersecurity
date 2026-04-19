"""
Test completo del flusso RAG: da dataset originale a embedding + indice FAISS.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("dataset").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from dataset import Dataset
from text_dataset import TextDataset
from embedding import Embedding
from vector_index import VectorIndex

def main():
    print("=== Test flusso RAG con FAISS ===\n")

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
    print(f"   Train: {len(train_text.get_texts())} testi, Test: {len(test_text.get_texts())} testi")

    print("6. Creazione modello di embedding...")
    emb = Embedding()
    print("   Modello caricato (all-MiniLM-L6-v2)")

    print("7. Generazione embedding per il training set...")
    train_embeddings = train_text.text_to_emb(emb)
    print(f"   Shape embedding: {train_embeddings.get_embedding().shape}")

    print("8. Generazione embedding per il test set...")
    test_embeddings = test_text.text_to_emb(emb)
    print(f"   Shape embedding: {test_embeddings.get_embedding().shape}")

    print("9. Distribuzione classi:")
    print("   Train:", train_text.get_targets().value_counts().to_dict())
    print("   Test:", test_text.get_targets().value_counts().to_dict())

    print("10. Creazione indice FAISS sul training set...")
    index = VectorIndex()
    index.build(train_embeddings, texts=train_text.get_texts(), metric="L2")
    index.save("faiss_index")
    print("   Indice costruito e salvato.")

    print("11. Caricamento indice da disco...")
    index_loaded = VectorIndex()
    index_loaded.load("faiss_index")
    print(f"   Indice caricato, tipo: {index_loaded.get_index_type()}, dimensione: {index_loaded._dimension}")

    print("12. Ricerca dei top-3 vicini per il primo embedding del test set...")
    query_emb = test_embeddings.get_embedding()[0]
    distances, indices = index_loaded.search(query_emb, k=3)
    print(f"   Indici trovati: {indices}")
    print(f"   Distanze: {[f'{d:.6f}' for d in distances]}")

    retrieved = index_loaded.get_metadata_by_indices(indices.tolist())
    print("   Testi e target dei vicini:")
    for i, (text, target) in enumerate(retrieved):
        print(f"     {i+1}: target={target}, testo={text[:100]}...")

    os.remove(sample_file)
    print("\n✅ Flusso RAG completato con successo!")

if __name__ == "__main__":
    main()