# RAG per Cybersecurity

### Retrieval-Augmented Generation per la classificazione di file Windows PE (malware vs goodware)

---

## Panoramica

Questo progetto implementa un sistema di **Retrieval-Augmented Generation (RAG)** per la classificazione di file **Windows Portable Executable (PE)** in due categorie: **malware** e **goodware**.

La pipeline sviluppata integra:

- Selezione statistica delle feature tramite Mutual Information (MI)
- Trasformazione testuale di dati strutturati (`feature: valore`)
- Generazione di embedding semantici con SentenceTransformer (`all-MiniLM-L6-v2`)
- Ricerca efficiente di similarità tramite FAISS (distanza L2)
- Supporto a due modelli LLM:
  - **DistilGPT2** (causale)
  - **Qwen3-4B-Instruct** (chat, con quantizzazione 4-bit)
- Tracciamento dei token dei vicini per monitorare troncamenti e ottimizzare il contesto

L’obiettivo è valutare l’efficacia di un approccio RAG su feature statiche di malware, confrontando due diverse architetture di LLM e una baseline di majority voting (k-NN).

---

## 🎯 Obiettivi

- Trasformare vettori di feature PE ad alta dimensionalità (2381 feature) in rappresentazioni testuali interpretabili
- Applicare la **Mutual Information (MI)** per la selezione delle feature più rilevanti
- Generare embedding densi tramite modelli Transformer
- Costruire un sistema di retrieval basato su similarità con FAISS
- Supportare la classificazione tramite LLM (con fallback a majority voting)
- Tracciare la lunghezza in token dei prompt e dei vicini

---

## 📚 Riferimento al paper OLIVANDER

Il set di 2381 feature statiche deriva dall’articolo:

> **"OLIVANDER: A Framework for Portable Executable Malware Detection Using Static Features"**  
> *A. N. Jahromi et al. – CCS '21*

Le feature comprendono:

- Istogramma ed entropia dei byte (256 + 256)
- Statistiche sulle stringhe stampabili (104)
- Attributi generali del file (10)
- Header COFF/Optional (62)
- Sezioni (255)
- Import/export table (1280 + 128)
- Data directory (30)

**Nota:** i file CSV (`BODMAS_features.csv`, `test.csv`) provengono dal dataset **BODMAS**.

---

## ⚙️ Requisiti

- Python **3.11+**
- pip

---

## 🚀 Installazione

### Clonazione repository

```bash
git clone https://github.com/FrancescoGassi/RagCybersecurity.git
cd RAG_Cybersecurity
````

### Ambiente virtuale (consigliato)

```bash
python -m venv .venv
```

Attivazione:

* Windows:

  ```bash
  .venv\Scripts\activate
  ```
* Linux/macOS:

  ```bash
  source .venv/bin/activate
  ```

### Installazione dipendenze

```bash
pip install -r requirements.txt
```

> ⚠️ Su Windows, sostituire `faiss-gpu` con `faiss-cpu`.

---

## 📂 Struttura del Progetto

```
RAG_Cybersecurity/
├── docs/
│   └── DiagrammaClassi.png
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── text_dataset.py
│   ├── embedding_dataset.py
│   ├── embedding.py
│   ├── vector_index.py
│   ├── llm_predictor.py
│   └── feature_names.py
│
├── create_named_csv.py
├── test_feature_count.py
├── test_ml.py
├── test_rag_flow.py
├── run_rag.py
├── run_majority_voting.py
├── requirements.txt
└── README.md
```

---

## 📊 Preparazione Dati

Inserire i file in:

```
DatasetPE/
```

File richiesti:

* `BODMAS_features.csv`
* `test.csv`

### Generazione file con intestazioni

```bash
python create_named_csv.py
```

Output:

* `BODMAS_features_named.csv`
* `test_named.csv`

---

## 🧪 Test

### Verifica feature

```bash
python test_feature_count.py
```

Output atteso:

```
Conteggio feature corretto (2381)
```

### Test Dataset

```bash
python test_ml.py
```

### Test pipeline RAG

```bash
python test_rag_flow.py
```

---

## 🚀 Esperimenti

### Baseline (k-NN)

```bash
python run_majority_voting.py
```

Output:

* `majority_voting_predictions.csv`

---

### RAG con DistilGPT2

Impostare:

```python
USE_QWEN = False
```

Eseguire:

```bash
python run_rag.py
```

Output:

* `rag_predictions_distilgpt2.csv`

---

### RAG con Qwen

Impostare:

```python
USE_QWEN = True
MAX_TEST_SAMPLES = 500
```

Eseguire:

```bash
python run_rag.py
```

Output:

* `rag_predictions_qwen.csv`

> ⚠️ Tempo medio: ~6-7 secondi per campione su GPU T4

---

## 🧠 Token Tracking

Colonne aggiuntive nei risultati:

* `total_prompt_tokens`
* `neighbor_tokens`
* `truncated`

Permettono di analizzare:

* Troncamenti
* Lunghezza contesto
* Ottimizzazione `MAX_TOKENS`

---

## 🔧 Configurazione (`src/config.py`)

```python
LLM_MODEL_NAME = "DistilGPT2"
MODEL_TYPE = "causal"
MAX_TOKENS = 1024
K_NEIGHBORS = 5
SAMPLE_SIZE = None
QWEN_USE_4BIT = False
```

---

## 🧹 Pulizia

```bash
rm -rf cache/
rm -f *.pkl *.faiss *_predictions.csv
```

---
