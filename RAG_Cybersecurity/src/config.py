from __future__ import annotations

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"   # File di training con intestazioni
TEST_CSV = "DatasetPE/test_named.csv"               # File di test con intestazioni


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
CACHE_DIR = "cache_rag_faiss_mv"                    # Directory per la cache FAISS
OUTPUT_CSV = "rag_predictions.csv"                  # Output della predizione con LLM
MAJORITY_VOTING_OUTPUT_CSV = "majority_voting_predictions.csv"  # Output della predizione con MV


# ------------------------------------------------------------
# Campionamento (dimensioni dei dataset)
# ------------------------------------------------------------
RANDOM_SEED = 42                                    # Seed per riproducibilità
TRAIN_SAMPLE_SIZE = 5000                            # Numero campioni training (None = tutto)
TEST_LIMIT = None                                   # Numero campioni test (None = tutto)
BALANCE_TRAINING = True                             # Bilancia le classi nel training
BALANCE_TEST = True                                 # Bilancia le classi nel test


# ------------------------------------------------------------
# Feature e retrieval
# ------------------------------------------------------------
TOP_K_FEATURES = 64                                 # Numero di feature selezionate con MI
K_NEIGHBORS = 7                                     # Numero di vicini FAISS (dispari)


# ------------------------------------------------------------
# Prompt per LLM
# ------------------------------------------------------------
MAX_EXAMPLES_IN_PROMPT = 5                          # Massimo numero di esempi mostrati
MAX_FEATURES_IN_PROMPT = 5                          # Massimo numero di feature mostrate
MIN_FEATURES_IN_PROMPT = 3                          # Minimo feature mostrate (riduzione)
MAX_PROMPT_TOKENS = 1024                            # Limite token per il prompt


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"         # Modello Instruct piccolo
USE_4BIT = True                                     # Quantizzazione a 4 bit (se supportata)
MAX_NEW_TOKENS = 4                                  # Token generati (basta per GOODWARE/MALWARE)
DEBUG_LLM = True                                    # Stampa output grezzi e parsing


# ------------------------------------------------------------
# Consenso semantico (varianti di prompt)
# ------------------------------------------------------------
PROMPT_VARIANTS = 3                                 # Numero di varianti di prompt (1-3)
MIN_CONSISTENT_LLM_VOTES = 2                        # Voti minimi concordi per avere candidato LLM


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------
CACHE_VERSION = "rag-faiss-semantic-3variants-v1"   # Versione della cache
FORCE_REBUILD_INDEX = False                         # Ricostruisce l'indice FAISS a ogni esecuzione