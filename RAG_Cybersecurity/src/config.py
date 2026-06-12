# Nome del modello LLM su Hugging Face
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# Quantizzazione 4-bit (True per ridurre memoria e accelerare)
USE_4BIT = False

# Se True, stampa output del modello (utile solo per debug)
DEBUG_LLM = True

# Campionamento dei dati (None = tutto il training/test)
SAMPLE_SIZE = 20
TEST_LIMIT = 20

# Parametri del modello RAG
MODEL_TYPE = "chat"             # Template chat di transformers
MAX_TOKENS = 2048               # Token massimi per il prompt
K_NEIGHBORS = 10                # Numero di vicini da recuperare con FAISS