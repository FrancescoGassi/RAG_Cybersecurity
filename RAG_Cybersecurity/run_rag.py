import sys
import os
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MAX_TOKENS, K_NEIGHBORS, SAMPLE_SIZE

USE_QWEN = False   # False = DistilGPT2, True = Qwen/Qwen3-4B-Instruct-2507

if USE_QWEN:
    LLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    MODEL_TYPE = "chat"
    QWEN_USE_4BIT = True   # Se hai poca VRAM, lascia True; se hai tanta VRAM, metti False
else:
    LLM_MODEL_NAME = "DistilGPT2"
    MODEL_TYPE = "causal"
    QWEN_USE_4BIT = False

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

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def print_step(step_text):
    print(f"\n{step_text}...")

def main():
    print(f"=== RAG (con LLM) ===")

    # 1. Caricamento dataset
    print_step("1. Caricamento training e test")
    train_ds = Dataset('DatasetPE/BODMAS_features_named.csv')
    test_ds  = Dataset('DatasetPE/test_named.csv')
    print(f"   Training: {len(train_ds)} esempi, {len(train_ds.feature_names)} feature")
    print(f"   Test:     {len(test_ds)} esempi, {len(test_ds.feature_names)} feature")

    if SAMPLE_SIZE and len(train_ds) > SAMPLE_SIZE:
        df_temp = train_ds.feat_data.copy()
        df_temp['__target__'] = train_ds.target_data
        df_sample = df_temp.sample(n=SAMPLE_SIZE, random_state=42)
        train_ds.feat_data = df_sample.drop('__target__', axis=1)
        train_ds.target_data = df_sample['__target__']
        print(f"   → Usato sottocampione del training: {SAMPLE_SIZE} esempi")
    else:
        print(f"   → Usato training completo ({len(train_ds)} esempi)")

    # 2. Mutual Information
    print_step("2. Calcolo Mutual Information (sul training)")
    with tqdm(total=1, desc="   Calcolo MI", bar_format="{l_bar}{bar}"):
        mi = train_ds.compute_mutual_information()
    top5 = list(mi.keys())[:5]
    print(f"   Top-5 feature (dal training): {', '.join(top5)}")

    # 3. Ordinamento feature
    print_step("3. Ordinamento feature secondo MI")
    train_sorted = train_ds.sort_features_by_mi(mi)
    test_sorted  = test_ds.sort_features_by_mi(mi)
    print(f"   Training ordinato: {len(train_sorted)} esempi")
    print(f"   Test ordinato:     {len(test_sorted)} esempi")

    # 4. Conversione in testo (usa cache separata per evitare conflitti tra modelli)
    model_suffix = "_qwen" if USE_QWEN else "_distilgpt2"
    train_pkl = os.path.join(CACHE_DIR, f'train_texts{model_suffix}.pkl')
    test_pkl = os.path.join(CACHE_DIR, f'test_texts{model_suffix}.pkl')
    if not os.path.exists(train_pkl):
        train_sorted.save(train_pkl)
    if not os.path.exists(test_pkl):
        test_sorted.save(test_pkl)
    train_text = TextDataset(train_pkl)
    test_text = TextDataset(test_pkl)
    print("   Testi salvati/ricaricati (in cache/)")

    # 5. Embedding + FAISS
    print_step("5. Generazione embedding e indice FAISS (L2)")
    emb_model = Embedding()
    print("   Creazione embedding del training set...")
    train_emb = train_text.text_to_emb(emb_model)

    index_prefix = os.path.join(CACHE_DIR, f"faiss_index{model_suffix}")
    if not os.path.exists(index_prefix + ".faiss"):
        index = VectorIndex()
        index.build(train_emb, texts=train_text.get_texts(), metric="L2")
        index.save(index_prefix)
    index_loaded = VectorIndex()
    index_loaded.load(index_prefix)
    print(f"   Indice FAISS caricato (dimensione {index_loaded._dimension})")

    # 6. LLM
    print_step(f"6. Caricamento modello LLM: {LLM_MODEL_NAME}")
    llm = LLMPredictor(max_tokens=MAX_TOKENS, model_type=MODEL_TYPE)
    llm.load(model_name=LLM_MODEL_NAME, qwen_use_4bit=QWEN_USE_4BIT)

    # 7. Valutazione RAG
    print_step(f"7. Valutazione RAG (k={K_NEIGHBORS})")
    output_csv = f"rag_predictions{model_suffix}.csv"
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
    acc = (y_true == y_pred).mean()

    print("\n" + "=" * 60)
    model_name_short = "Qwen3-4B" if USE_QWEN else "DistilGPT2"
    title = f"RAG con {model_name_short} – RISULTATI"
    print(" " * ((60 - len(title)) // 2) + title)
    print("=" * 60)
    print(f"\nACCURATEZZA: {acc*100:.2f}%")
    print("\nMATRICE DI CONFUSIONE:")
    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=['goodware', 'malware'], columns=['pred_goodware', 'pred_malware']))
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['goodware', 'malware']))
    
    # Statistiche token
    print("\nSTATISTICHE TOKEN:")
    print(f"Prompt tokens - media: {predictions_df['total_prompt_tokens'].mean():.1f}, max: {predictions_df['total_prompt_tokens'].max()}")
    print(f"Predizioni troncate: {predictions_df['truncated'].sum()} / {len(predictions_df)}")
    neighbor_len = predictions_df['neighbor_tokens'].apply(lambda s: len(eval(s)) if s != '[]' else 0).mean()
    print(f"Numero medio di vicini effettivi (con token>0): {neighbor_len:.1f}")
    
    print("=" * 60)
    print(f"\n✅ Completato. CSV salvato in {output_csv}")

if __name__ == "__main__":
    main()