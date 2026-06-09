# Nome del modello LLM su Hugging Face
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Quantizzazione 4-bit (True per ridurre memoria e accelerare)
USE_4BIT = True

# Se True, stampa output del modello (utile solo per debug)
DEBUG_LLM = False

# Campionamento dei dati (None = tutto il training/test)
SAMPLE_SIZE = None
TEST_LIMIT = None

# Parametri del modello RAG
MODEL_TYPE = "chat"             # Template chat di transformers
MAX_TOKENS = 4096               # Token massimi per il prompt
K_NEIGHBORS = 5                 # Numero di vicini da recuperare con FAISS