import sys
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Aggiunge la directory padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SAMPLE_SIZE, K_NEIGHBORS, TEST_LIMIT

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
for lib in ["huggingface_hub", "sentence_transformers", "transformers", "faiss", "torch"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

from src.dataset import Dataset
from src.text_dataset import TextDataset
from src.embedding import Embedding
from src.vector_index import VectorIndex

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def print_step(step_text):
    print(f"\n{step_text}...")

def print_experiment_params():
    print("\n" + "=" * 70)
    print(" PARAMETRI ESPERIMENTO - MAJORITY VOTING (k-NN) ".center(70))
    print("=" * 70)
    print(f"  Campioni training      : {SAMPLE_SIZE if SAMPLE_SIZE else 'TUTTI'}")
    print(f"  Campioni test          : {TEST_LIMIT if TEST_LIMIT else 'TUTTI'}")
    print(f"  Numero vicini (k)      : {K_NEIGHBORS}")
    print("  Modello LLM             : NON UTILIZZATO")
    print("=" * 70)

def main():
    print_experiment_params()
    print("\n=== MAJORITY VOTING (k-NN senza LLM) ===")

    # 1. Caricamento dati
    print_step("1. Caricamento dataset")
    train_ds = Dataset('DatasetPE/BODMAS_features_named.csv')
    test_ds  = Dataset('DatasetPE/test_named.csv')
    print(f"   Training originale: {len(train_ds)} esempi, {len(train_ds.feature_names)} feature")
    print(f"   Test originale:     {len(test_ds)} esempi, {len(test_ds.feature_names)} feature")

    if SAMPLE_SIZE and len(train_ds) > SAMPLE_SIZE:
        df_temp = train_ds.feat_data.copy()
        df_temp['__target__'] = train_ds.target_data
        df_sample = df_temp.sample(n=SAMPLE_SIZE, random_state=42)
        train_ds.feat_data = df_sample.drop('__target__', axis=1)
        train_ds.target_data = df_sample['__target__']
        print(f"   → Training ridotto a {SAMPLE_SIZE} esempi")

    # 2. Mutual Information
    print_step("2. Calcolo Mutual Information (sul training)")
    print("   Calcolo MI in corso...", end=' ', flush=True)
    mi = train_ds.compute_mutual_information(sample_size=SAMPLE_SIZE)
    print("completato.")
    top5 = list(mi.keys())[:5]
    print(f"   Top-5 feature: {', '.join(top5)}")

    # 3. Ordinamento feature
    print_step("3. Ordinamento delle feature per MI")
    train_sorted = train_ds.sort_features_by_mi(mi, top_k=None)
    test_sorted  = test_ds.sort_features_by_mi(mi, top_k=None)

    # Conversione in testo - cache
    train_pkl = os.path.join(CACHE_DIR, 'train_texts.pkl')
    test_pkl = os.path.join(CACHE_DIR, 'test_texts.pkl')
    if not os.path.exists(train_pkl):
        train_sorted.save(train_pkl)
    if not os.path.exists(test_pkl):
        test_sorted.save(test_pkl)
    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)
    print("   Testi salvati/ricaricati in cache/ (train_texts.pkl, test_texts.pkl)")

    # Limitazione test set (bilanciata)
    if TEST_LIMIT is not None:
        targets = test_text.get_targets().tolist()
        indices = np.arange(len(targets))
        if len(np.unique(targets)) == 2 and min(pd.Series(targets).value_counts()) >= TEST_LIMIT // 2:
            _, sampled_idx = train_test_split(indices, test_size=TEST_LIMIT, stratify=targets, random_state=42)
        else:
            sampled_idx = indices[:TEST_LIMIT]
        test_texts = [test_text.get_texts()[i] for i in sampled_idx]
        true_labels_num = [targets[i] for i in sampled_idx]
        print(f"   → Test limitato a {len(test_texts)} campioni (bilanciati: {pd.Series(true_labels_num).value_counts().to_dict()})")
    else:
        test_texts = test_text.get_texts()
        true_labels_num = test_text.get_targets().tolist()
        print(f"   → Test completo ({len(test_texts)} campioni)")

    # 4. Embedding e indice FAISS con costruzione incrementale (metrica L2)
    print_step("4. Generazione embedding e indice FAISS (metrica L2)")
    emb_model = Embedding()
    index_prefix = os.path.join(CACHE_DIR, "faiss_index_mv")
    if not os.path.exists(index_prefix + ".faiss"):
        print("   Costruzione indice incrementale da testi...")
        index = VectorIndex()
        index.build_from_texts(train_text, emb_model, metric="L2", batch_size=64)
        index.save(index_prefix)
    index_loaded = VectorIndex()
    index_loaded.load(index_prefix)
    print(f"   Indice FAISS caricato (dimensione {index_loaded._dimension}, metrica {index_loaded.get_index_type()})")

    # 5.Predizione
    print_step(f"5. Predizione con Majority Voting (k={K_NEIGHBORS})")
    results = []
    for query, true_label_num in tqdm(zip(test_texts, true_labels_num), total=len(test_texts), desc="   Progresso"):
        q_emb = emb_model.encode([query])[0]
        _, indices = index_loaded.search(q_emb, k=K_NEIGHBORS)
        _, ret_targets = index_loaded.get_metadata_by_indices(indices)
        counts = np.bincount(ret_targets)
        pred_num = int(np.argmax(counts))
        pred_str = "malware" if pred_num == 1 else "goodware"
        true_label_str = "malware" if true_label_num == 1 else "goodware"
        results.append({'prediction': pred_str, 'true_label': true_label_str})

    df = pd.DataFrame(results)
    output_csv = "majority_voting_predictions.csv"
    df.to_csv(output_csv, index=False)
    acc = (df['prediction'] == df['true_label']).mean()
    print(f"\n   Accuratezza: {acc*100:.2f}%")

    # Report finale
    y_true = df['true_label']
    y_pred = df['prediction']
    print("\n" + "=" * 70)
    print(" RISULTATI ESPERIMENTO - MAJORITY VOTING (k-NN) ".center(70, "="))
    print("=" * 70)
    print(f"\nACCURATEZZA: {acc*100:.2f}%")
    print("\nMATRICE DI CONFUSIONE:")
    cm = confusion_matrix(y_true, y_pred, labels=['goodware', 'malware'])
    print(pd.DataFrame(cm, index=['goodware', 'malware'], columns=['pred_goodware', 'pred_malware']))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['goodware', 'malware']))
    print("=" * 70)
    print(f"\n✅ CSV salvato in {output_csv}")

if __name__ == "__main__":
    main()