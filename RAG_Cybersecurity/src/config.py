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
TRAIN_SAMPLE_SIZE = 5000
TEST_LIMIT = 20
BALANCE_TRAINING = True
BALANCE_TEST = True

# Feature e retrieval
TOP_K_FEATURES = 64
K_NEIGHBORS = 7

# Prompt
MAX_EXAMPLES_IN_PROMPT = 5
MAX_FEATURES_IN_PROMPT = 5
MIN_FEATURES_IN_PROMPT = 3
MAX_PROMPT_TOKENS = 1024

# LLM
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
USE_4BIT = True
MAX_NEW_TOKENS = 4
DEBUG_LLM = True

# Consenso semantico
PROMPT_VARIANTS = 3
MIN_CONSISTENT_LLM_VOTES = 2

# Cache
CACHE_VERSION = "rag-faiss-semantic-3variants-v1"
FORCE_REBUILD_INDEX = False