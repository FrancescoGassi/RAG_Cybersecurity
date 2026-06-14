from __future__ import annotations

# Dataset
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"
TEST_CSV = "DatasetPE/test_named.csv"

# Output
CACHE_DIR = "cache_rag_faiss_mv"
OUTPUT_CSV = "rag_predictions.csv"
MAJORITY_VOTING_OUTPUT_CSV = "majority_voting_predictions.csv"

# Campionamento
RANDOM_SEED = 42
TRAIN_SAMPLE_SIZE = 2000      # None = tutto il training set
TEST_LIMIT = 20               # None = tutto il test set
BALANCE_TRAINING = True
BALANCE_TEST = True

# Feature selection
TOP_K_FEATURES = 20

# FAISS / Majority Voting
K_NEIGHBORS = 7               # dispari: evita pareggi nel MV

MAX_EXAMPLES_IN_PROMPT = 5    # massimo vicini mostrati nel prompt (poi bilanciati)
MAX_FEATURES_IN_PROMPT = 5
MIN_FEATURES_IN_PROMPT = 4
MAX_PROMPT_TOKENS = 2048      # aumentato per descrizioni e bilanciamento

# LLM
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"   # puoi cambiare con "Qwen/Qwen2-1.5B-Instruct"
USE_4BIT = True               # usato solo se CUDA disponibile
MAX_NEW_TOKENS = 4
DEBUG_LLM = True

PROMPT_VARIANTS = 2           # due varianti per consenso
MIN_CONSISTENT_LLM_VOTES = 2

# Se FAISS è molto netto e l'LLM contraddice, usa MV.
USE_STRONG_MV_GUARD = True
STRONG_MV_MIN_MARGIN = 5      # con k=7: 6-1 o 7-0

# Cache
CACHE_VERSION = "rag-faiss-natural-neighbors-v5"   # versione incrementata
FORCE_REBUILD_INDEX = False