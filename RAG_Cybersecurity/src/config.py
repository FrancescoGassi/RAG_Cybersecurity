# Configurazione

LLM_MODEL_NAME = "DistilGPT2"          # Default, verrà sovrascritto da run_rag.py
MODEL_TYPE = "causal"                  # Default, verrà sovrascritto
MAX_TOKENS = 1024                      # Token massimi per il prompt LLM
K_NEIGHBORS = 5                        # Numero di vicini per k-NN / RAG
SAMPLE_SIZE = None                     # None = usa tutto il training set

QWEN_USE_4BIT = False                  # Default, verrà sovrascritto