# Nome del modello LLM su Hugging Face
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Quantizzazione 4-bit (True solo per modelli large e se disponi di GPU)
QWEN_USE_4BIT = True

# Se True, stampa nella console l'output generato dal modello (utile per debug)
DEBUG_LLM = True

# Campionamento dei dati
SAMPLE_SIZE = None        # Numero di campioni per il training (None = tutto il training)
TEST_LIMIT = None         # Numero di campioni per il test (None = tutto il test)

# Parametri del modello RAG
MODEL_TYPE = "chat"     # template chat di transformers
MAX_TOKENS = 4096       # Token massimi per il prompt
K_NEIGHBORS = 5         # Numero di vicini da recuperare con FAISS