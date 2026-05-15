# Configurazione

# Modello LLM predefinito (Qwen2-0.5B-Instruct)
LLM_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MODEL_TYPE = "chat"                  # Qwen è un modello chat
MAX_TOKENS = 1024                    # Token massimi per il prompt LLM
K_NEIGHBORS = 5                      # Numero di vicini per k-NN / RAG
SAMPLE_SIZE = None                   # None = usa tutto il training set

# Quantizzazione 4-bit (solo per modelli grandi, es. Qwen3-4B)
QWEN_USE_4BIT = False                # Per il modello piccolo (0.5B) non serve