from __future__ import annotations

# Dataset
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"
TEST_CSV = "DatasetPE/test_named.csv"
TARGET_COLUMN: str | None = None                # None = ultima colonna

# Campionamento
RANDOM_SEED = 42
TRAIN_SAMPLE_SIZE: int | None = None
TEST_LIMIT: int | None = None
BALANCE_TRAINING = False
BALANCE_TEST = False

# Selezione feature e retrieval
TOP_K_FEATURES: int = 128
K_NEIGHBORS = 3

# Majority voting
PURE_MAJORITY_VOTING = True                     # True = solo conteggio delle etichette; False = ibrido (con similarità)
HYBRID_MAJORITY_WEIGHT = 0.50                   # peso per il conteggio (usato solo se PURE_MAJORITY_VOTING è False)

# Embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128
EMBEDDING_SAMPLE_BATCH_SIZE = 128
EMBEDDING_TOP_FEATURES_PER_SAMPLE = 128
EMBEDDING_FEATURES_PER_CHUNK = 8
EMBEDDING_DEVICE: str | None = None             # None, "cpu" o "cuda"
SHOW_PROGRESS = True

# --- MODALITÀ DI RETRIEVAL ---
# "semantic"  -> usa il sistema RAG con SentenceTransformer
# "raw"       -> usa i vettori numerici standardizzati direttamente (baseline k-NN)
RETRIEVAL_MODE = "raw"

# Cache e output
CACHE_DIR = "cache"
CACHE_NAME = "training_index"
FORCE_REBUILD_INDEX = True
OUTPUT_CSV = "majority_voting_embedding_predictions.csv"

# LLM
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEBUG_LLM = False
USE_4BIT = True
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 10
MAX_EXAMPLES_IN_PROMPT = 5
MAX_FEATURES_IN_PROMPT = 20
MIN_FEATURES_IN_PROMPT = 3
PROMPT_VARIANTS = 3
MIN_CONSISTENT_LLM_VOTES = 2