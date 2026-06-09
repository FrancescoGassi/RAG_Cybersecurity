import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.dataset import Dataset
from src.text_dataset import TextDataset
from src.embedding import Embedding
from src.vector_index import VectorIndex

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def print_step(step_text):
    print(f"\n{step_text}...")

def main():
    print("=== Test flusso RAG con FAISS ===")

    for f in ['train_texts.pkl', 'test_texts.pkl', 'faiss_index.faiss', 'faiss_index_metadata.pkl']:
        path = os.path.join(CACHE_DIR, f)
        if os.path.exists(path):
            os.remove(path)

    print_step("1. Caricamento campione del dataset")
    sample_size = 1000
    df = pd.read_csv('DatasetPE/BODMAS_features_named.csv')
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    sample_file = os.path.join(CACHE_DIR, 'temp_sample.csv')
    df_sample.to_csv(sample_file, index=False)

    ds = Dataset(sample_file)
    print(f"   Caricati {len(ds)} esempi, {len(ds.feature_names)} feature")

    print_step("2. Calcolo mutual information e ordinamento")
    mi = ds.compute_mutual_information()
    sorted_ds = ds.sort_features_by_mi(mi)
    print(f"   Top 5 feature: {', '.join(sorted_ds.feature_names[:5])}")

    print_step("3. Split in train/test")
    train_ds, test_ds = sorted_ds.train_test_split(test_size=0.2, random_state=42)
    print(f"   Train: {len(train_ds)} esempi, Test: {len(test_ds)} esempi")

    print_step("4. Serializzazione in formato testuale")
    train_pkl = os.path.join(CACHE_DIR, 'train_texts.pkl')
    test_pkl = os.path.join(CACHE_DIR, 'test_texts.pkl')
    train_ds.save(train_pkl)
    test_ds.save(test_pkl)
    print("   File salvati in cache/")

    print_step("5. Caricamento TextDataset")
    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)
    print(f"   Train: {len(train_text.get_texts())} testi, Test: {len(test_text.get_texts())} testi")

    print_step("6. Creazione modello di embedding")
    emb = Embedding()
    print("   Modello caricato (all-MiniLM-L6-v2)")

    print_step("7. Generazione embedding per il training set")
    train_embeddings = train_text.text_to_emb(emb)
    print(f"   Shape embedding: {train_embeddings.get_embedding().shape}")

    print_step("8. Generazione embedding per il test set")
    test_embeddings = test_text.text_to_emb(emb)
    print(f"   Shape embedding: {test_embeddings.get_embedding().shape}")

    print_step("9. Distribuzione classi")
    train_dist = train_text.get_targets().value_counts().to_dict()
    test_dist = test_text.get_targets().value_counts().to_dict()
    print(f"   Train: {train_dist}")
    print(f"   Test:  {test_dist}")

    print_step("10. Creazione indice FAISS sul training set")
    index = VectorIndex()
    index.build(train_embeddings, texts=train_text.get_texts(), metric="L2")
    index_path = os.path.join(CACHE_DIR, "faiss_index")
    index.save(index_path)
    print("   Indice costruito e salvato in cache/")

    print_step("11. Caricamento indice da disco")
    index_loaded = VectorIndex()
    index_loaded.load(index_path)
    print(f"   Indice caricato, tipo: {index_loaded.get_index_type()}, dimensione: {index_loaded._dimension}")

    print_step("12. Ricerca dei top-3 vicini per il primo embedding del test set")
    query_emb = test_embeddings.get_embedding()[0]
    distances, indices = index_loaded.search(query_emb, k=3)
    print(f"   Indici trovati: {indices}")
    print(f"   Distanze: {[f'{d:.6f}' for d in distances]}")

    ret_texts, ret_targets = index_loaded.get_metadata_by_indices(indices.tolist())
    print("   Testi e target dei vicini:")
    for i, (text, target) in enumerate(zip(ret_texts, ret_targets)):
        print(f"     {i+1}: target={target}, testo={text[:100]}...")

    os.remove(sample_file)
    print("\n✅ Flusso RAG completato con successo!")

if __name__ == "__main__":
    main()