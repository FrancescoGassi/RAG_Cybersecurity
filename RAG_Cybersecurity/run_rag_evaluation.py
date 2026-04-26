import sys
import os
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
for lib in ["huggingface_hub", "httpx", "sentence_transformers", "transformers", "urllib3", "faiss", "torch", "filelock", "datasets"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

sys.path.insert(0, 'src')

from src.dataset import Dataset
from src.text_dataset import TextDataset
from src.embedding import Embedding
from src.vector_index import VectorIndex
from src.llm_predictor import LLMPredictor

# ===== CONFIGURAZIONE =====
SAMPLE_SIZE = 2000
K_NEIGHBORS = 5
MODEL_NAME = "distilgpt2"
MAX_TOKENS = 512
# =========================

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def print_step(step_text):
    print(f"\n{step_text}...")

def main():
    print("=== RAG per malware detection ===")

    print_step("1. Caricamento dataset")
    csv_path = 'DatasetPE/BODMAS_features_named.csv'
    ds = Dataset(csv_path)
    print(f"   Totale originale: {len(ds)} esempi, {len(ds.feature_names)} feature")

    if SAMPLE_SIZE and len(ds) > SAMPLE_SIZE:
        df_temp = ds.feat_data.copy()
        df_temp['__target__'] = ds.target_data
        df_sample = df_temp.sample(n=SAMPLE_SIZE, random_state=42)
        ds.feat_data = df_sample.drop('__target__', axis=1)
        ds.target_data = df_sample['__target__']
        print(f"   → Usato campione di {SAMPLE_SIZE} esempi")

    print_step("2. Calcolo Mutual Information")
    with tqdm(total=1, desc="   Calcolo MI", bar_format="{l_bar}{bar}") as pbar:
        mi = ds.compute_mutual_information()
        pbar.update(1)
    top5 = list(mi.keys())[:5]
    print(f"   Top-5 feature: {', '.join(top5)}")

    print_step("3. Split train/test (80/20 stratificato)")
    sorted_ds = ds.sort_features_by_mi(mi)
    train_ds, test_ds = sorted_ds.train_test_split(test_size=0.2, random_state=42)
    print(f"   Train: {len(train_ds)} esempi")
    print(f"   Test:  {len(test_ds)} esempi")

    print_step("4. Conversione in formato testuale")
    train_pkl = os.path.join(CACHE_DIR, 'train_texts.pkl')
    test_pkl = os.path.join(CACHE_DIR, 'test_texts.pkl')
    if not os.path.exists(train_pkl):
        train_ds.save(train_pkl)
    if not os.path.exists(test_pkl):
        test_ds.save(test_pkl)
    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)
    print("   Testi salvati/ricaricati (in cache/)")

    print_step("5. Generazione embedding e indice FAISS (L2)")
    emb_model = Embedding()
    print("   Creazione embedding del training set...")
    train_emb = train_text.text_to_emb(emb_model)

    index_prefix = os.path.join(CACHE_DIR, "faiss_index")
    index = VectorIndex()
    index.build(train_emb, texts=train_text.get_texts(), metric="L2")
    index.save(index_prefix)
    index_loaded = VectorIndex()
    index_loaded.load(index_prefix)
    print(f"   Indice FAISS creato (dimensione {index_loaded._dimension})")

    print_step(f"6. Caricamento modello LLM: {MODEL_NAME}")
    llm = LLMPredictor(max_tokens=MAX_TOKENS)
    llm.load(MODEL_NAME)

    print_step(f"7. Valutazione RAG (k={K_NEIGHBORS})")
    output_csv = "rag_predictions.csv"
    predictions_df = llm.predict(
        test_texts=test_text.get_texts(),
        true_labels=test_text.get_targets().tolist(),
        vector_index=index_loaded,
        embedding_model=emb_model,
        k=K_NEIGHBORS,
        output_csv=output_csv
    )

    y_true = predictions_df['true_label']
    y_pred = predictions_df['prediction']
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['goodware', 'malware']))

    print(f"\n✅ Completato. CSV salvato in {output_csv}")

if __name__ == "__main__":
    main()