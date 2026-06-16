from __future__ import annotations

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
TRAIN_CSV = "DatasetPE/BODMAS_features_named.csv"
TEST_CSV = "DatasetPE/test_named.csv"
TARGET_COLUMN: str | None = None  # None = ultima colonna del CSV

# Verifica che nomi e ordine delle 2381 feature coincidano con BODMAS.
VALIDATE_BODMAS_FEATURE_NAMES = True

# Controlli contro una valutazione artificialmente ottimistica.
CHECK_TRAIN_TEST_OVERLAP = True
FAIL_ON_TRAIN_TEST_OVERLAP = False

# ------------------------------------------------------------
# Output e cache
# ------------------------------------------------------------
CACHE_DIR = "cache_rag_faiss_mv_embedding"
MAJORITY_VOTING_OUTPUT_CSV = "majority_voting_embedding_predictions.csv"
MAJORITY_VOTING_METRICS_JSON = "majority_voting_embedding_metrics.json"

# ------------------------------------------------------------
# Campionamento
# ------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_SAMPLE_SIZE: int | None = None  # None = intero training set
TEST_LIMIT: int | None = None         # None = intero test set
BALANCE_TRAINING = False
BALANCE_TEST = False

# ------------------------------------------------------------
# Selezione delle feature
# ------------------------------------------------------------
TOP_K_FEATURES = 128
MUTUAL_INFORMATION_N_NEIGHBORS = 3

# ------------------------------------------------------------
# Retrieval e voto
# ------------------------------------------------------------
K_NEIGHBORS = 3

# "majority" mantiene il Majority Voting classico.
# "similarity_weighted" somma le similarità positive per classe.
VOTING_STRATEGY = "majority"

# ------------------------------------------------------------
# Modello di embedding
# ------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Testi elaborati contemporaneamente dal SentenceTransformer.
# 64 è più prudente di 512 e riduce il rischio di out-of-memory.
EMBEDDING_BATCH_SIZE = 64

# Campioni PE trasformati in embedding per blocco operativo.
EMBEDDING_SAMPLE_BATCH_SIZE = 128

# Per ogni campione usa le feature con maggiore valore standardizzato assoluto.
# Il valore effettivo è min(questo parametro, TOP_K_FEATURES).
EMBEDDING_TOP_FEATURES_PER_SAMPLE = 32

# Un numero contenuto evita la troncatura silenziosa del tokenizer.
EMBEDDING_FEATURES_PER_CHUNK = 8
EMBEDDING_VALUE_DECIMALS = 4

# False mantiene l'ordine globale delle feature selezionate e rende la rappresentazione più stabile tra campioni diversi.
EMBEDDING_SORT_FEATURES_BY_MAGNITUDE = False

# Se True, l'esecuzione si interrompe anziché troncare testi troppo lunghi.
EMBEDDING_ENFORCE_MAX_LENGTH = True
EMBEDDING_SHOW_PROGRESS = True

# None = CUDA se disponibile, altrimenti CPU. Valori: None, "cpu", "cuda".
EMBEDDING_DEVICE: str | None = None

# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------
CACHE_VERSION = "mv-sentence-embedding-v2"
FORCE_REBUILD_INDEX = True
