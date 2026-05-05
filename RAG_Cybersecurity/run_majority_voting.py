import sys
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.config import SAMPLE_SIZE, K_NEIGHBORS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
for lib in ["huggingface_hub", "sentence_transformers", "transformers", "faiss", "torch"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

sys.path.insert(0, 'src')

from src.dataset import Dataset
from src.text_dataset import TextDataset
from src.embedding import Embedding
from src.vector_index import VectorIndex

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def print_step(step_text):
    print(f"\n{step_text}...")

def main():
    print("=== MAJORITY VOTING (k-NN senza LLM) ===")

    # 1. Caricamento dataset di training e test
    print_step("1. Caricamento training e test")
    train_ds = Dataset('DatasetPE/BODMAS_features_named.csv')
    test_ds  = Dataset('DatasetPE/test_named.csv')
    print(f"   Training originale: {len(train_ds)} esempi, {len(train_ds.feature_names)} feature")
    print(f"   Test originale:     {len(test_ds)} esempi, {len(test_ds.feature_names)} feature")

    # Sottocampionamento del training set
    if SAMPLE_SIZE and len(train_ds) > SAMPLE_SIZE:
        df_temp = train_ds.feat_data.copy()
        df_temp['__target__'] = train_ds.target_data
        df_sample = df_temp.sample(n=SAMPLE_SIZE, random_state=42)
        train_ds.feat_data = df_sample.drop('__target__', axis=1)
        train_ds.target_data = df_sample['__target__']
        print(f"   → Usato sottocampione del training: {SAMPLE_SIZE} esempi")
    print(f"   Training finale: {len(train_ds)} esempi")
    print(f"   Test finale:     {len(test_ds)} esempi")

    # 2. Calcolo Mutual Information (solo sul training)
    print_step("2. Calcolo Mutual Information (sul training)")
    with tqdm(total=1, desc="   Calcolo MI", bar_format="{l_bar}{bar}"):
        mi = train_ds.compute_mutual_information()
    top5 = list(mi.keys())[:5]
    print(f"   Top-5 feature (dal training): {', '.join(top5)}")

    # 3. Ordinamento feature secondo MI (su entrambi i dataset)
    print_step("3. Ordinamento feature secondo MI")
    train_sorted = train_ds.sort_features_by_mi(mi)
    test_sorted  = test_ds.sort_features_by_mi(mi)   # stesso ordine delle feature
    print(f"   Training ordinato: {len(train_sorted)} esempi")
    print(f"   Test ordinato:     {len(test_sorted)} esempi")

    # 4. Conversione in formato testuale
    print_step("4. Conversione in formato testuale")
    train_pkl = os.path.join(CACHE_DIR, 'train_texts.pkl')
    test_pkl = os.path.join(CACHE_DIR, 'test_texts.pkl')
    if not os.path.exists(train_pkl):
        train_sorted.save(train_pkl)
    if not os.path.exists(test_pkl):
        test_sorted.save(test_pkl)
    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)
    print("   Testi salvati/ricaricati (in cache/)")

    # 5. Generazione embedding e indice FAISS (L2)
    print_step("5. Generazione embedding e indice FAISS (L2)")
    emb_model = Embedding()
    print("   Creazione embedding del training set...")
    train_emb = train_text.text_to_emb(emb_model)
    index_prefix = os.path.join(CACHE_DIR, "faiss_index")
    if not os.path.exists(index_prefix + ".faiss"):
        index = VectorIndex()
        index.build(train_emb, texts=train_text.get_texts(), metric="L2")
        index.save(index_prefix)
    index_loaded = VectorIndex()
    index_loaded.load(index_prefix)
    print(f"   Indice FAISS creato (dimensione {index_loaded._dimension})")

    # 6. Predizione con Majority Voting (k-NN)
    print_step(f"6. Predizione con Majority Voting (k={K_NEIGHBORS})")
    results = []
    test_texts = test_text.get_texts()
    true_labels = test_text.get_targets().tolist()
    for query, true_label in tqdm(zip(test_texts, true_labels), total=len(test_texts), desc="   Progresso", unit="campione", ncols=80):
        q_emb = emb_model.encode([query])[0]
        _, indices = index_loaded.search(q_emb, k=K_NEIGHBORS)
        _, ret_targets = index_loaded.get_metadata_by_indices(indices)
        counts = np.bincount(ret_targets)
        pred = int(np.argmax(counts))
        results.append({'prediction': pred, 'true_label': true_label})

    df = pd.DataFrame(results)
    output_csv = "majority_voting_predictions.csv"
    df.to_csv(output_csv, index=False)
    acc = (df['prediction'] == df['true_label']).mean()

    print("\n" + "=" * 60)
    print(" " * 15 + "MAJORITY VOTING – RISULTATI")
    print("=" * 60)
    print(f"\nACCURATEZZA: {acc*100:.2f}%")
    print("\nMATRICE DI CONFUSIONE:")
    cm = confusion_matrix(df['true_label'], df['prediction'])
    print(pd.DataFrame(cm, index=['goodware', 'malware'], columns=['pred_goodware', 'pred_malware']))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(df['true_label'], df['prediction'], target_names=['goodware', 'malware']))
    print("=" * 60)
    print(f"\n✅ Completato. CSV salvato in {output_csv}")

if __name__ == "__main__":
    main()