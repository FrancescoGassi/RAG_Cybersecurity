from __future__ import annotations

# Dataset
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"                   # Percorso del file CSV di training
TEST_CSV = "DatasetPE/test_named.csv"                               # Percorso del file CSV di test

# Output
CACHE_DIR = "cache_rag_faiss_mv"                                    # Cartella per salvare l'indice FAISS e la cache
OUTPUT_CSV = "rag_predictions.csv"                                  # File CSV con le predizioni RAG+LLM
MAJORITY_VOTING_OUTPUT_CSV = "majority_voting_predictions.csv"      # File CSV con sole predizioni MV

# Campionamento
RANDOM_SEED = 42                                                    # Seed per la riproducibilità
TRAIN_SAMPLE_SIZE = 2000                                            # Numero di campioni dal training (None = tutto)
TEST_LIMIT = 20                                                     # Numero di campioni dal test (None = tutto)
BALANCE_TRAINING = True                                             # Bilanciamento classi nel training
BALANCE_TEST = True                                                 # Bilanciamento classi nel test

# Feature selection
TOP_K_FEATURES = 20                                                 # Numero di feature con MI più alta da selezionare

# FAISS / Majority Voting
K_NEIGHBORS = 7                                                     # Numero di vicini

MAX_EXAMPLES_IN_PROMPT = 5                                          # Massimo vicini mostrati nel prompt (poi bilanciati)
MAX_FEATURES_IN_PROMPT = 5                                          # Massimo numero di feature per esempio nel prompt
MIN_FEATURES_IN_PROMPT = 4                                          # Minimo feature per esempio (se token limit)
MAX_PROMPT_TOKENS = 2048                                            # Token massimi per il prompt

# LLM
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"                         # Modello LLM
USE_4BIT = True                                                     # Caricamento in 4-bit se CUDA disponibile
MAX_NEW_TOKENS = 4                                                  # Token da generare
DEBUG_LLM = True                                                    # Stampa dettagli predizioni LLM

PROMPT_VARIANTS = 2                                                 # Numero varianti prompt
MIN_CONSISTENT_LLM_VOTES = 2                                        # Voti minimi consistenti tra varianti per accettare LLM

# Se FAISS è molto netto e l'LLM contraddice, usa MV.
USE_STRONG_MV_GUARD = True                                          # Abilita fallback a MV se margine FAISS alto
STRONG_MV_MIN_MARGIN = 5                                            # Margine minimo per attivare fallback

# Cache
CACHE_VERSION = "rag-faiss-natural-neighbors-v5"                    # Versione cache
FORCE_REBUILD_INDEX = False                                         # Forza ricostruzione indice anche se cache valida