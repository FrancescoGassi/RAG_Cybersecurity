from __future__ import annotations

# Dataset
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"
TEST_CSV = "DatasetPE/test_named.csv"
TARGET_COLUMN: str | None = None  # None = ultima colonna

# Campionamento
RANDOM_SEED = 42
TRAIN_SAMPLE_SIZE: int | None = None
TEST_LIMIT: int | None = None
BALANCE_TRAINING = False
BALANCE_TEST = False

# Selezione feature e retrieval
TOP_K_FEATURES = 128
K_NEIGHBORS = 3
HYBRID_MAJORITY_WEIGHT = 0.50  # 0.50 conteggio + 0.50 similarità

# Embedding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_SAMPLE_BATCH_SIZE = 128
EMBEDDING_TOP_FEATURES_PER_SAMPLE = 32
EMBEDDING_FEATURES_PER_CHUNK = 8
EMBEDDING_DEVICE: str | None = None  # None, "cpu" o "cuda"
SHOW_PROGRESS = True

# Cache e output
CACHE_DIR = "cache"
CACHE_NAME = "training_index"
FORCE_REBUILD_INDEX = True
OUTPUT_CSV = "majority_voting_embedding_predictions.csv"
