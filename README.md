# RAG per Cybersecurity

### Retrieval-Augmented Generation per la classificazione di file Windows PE (malware vs goodware)

---

## Panoramica

Questo progetto implementa un sistema di **Retrieval-Augmented Generation (RAG)** per la classificazione di file **Windows Portable Executable (PE)** in due categorie: **malware** e **goodware**.

La pipeline sviluppata integra:

* Selezione statistica delle feature
* Trasformazione testuale di dati strutturati
* Generazione di embedding semantici
* Ricerca efficiente di similarità tramite FAISS

L’obiettivo è dimostrare come il **retrieval semantico applicato a feature strutturate** possa supportare attività di analisi e classificazione in ambito cybersecurity.

---

## 🎯 Obiettivi

* Trasformare vettori di feature PE ad alta dimensionalità in rappresentazioni testuali interpretabili
* Applicare la **Mutual Information (MI)** per la selezione delle feature rilevanti
* Generare embedding densi tramite modelli Transformer
* Costruire un sistema di retrieval basato su similarità con FAISS
* Supportare la classificazione tramite analisi dei vicini più simili

---

## 📚 Riferimento al paper OLIVANDER

Il set di 2381 feature statiche utilizzato in questo progetto deriva dal lavoro descritto nell'articolo:

> **"OLIVANDER: A Framework for Portable Executable Malware Detection Using Static Features"**  
> *Autori: A. N. Jahromi et al.*  
> Pubblicato in: *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security (CCS '21)*.

Le feature sono state estratte tramite la libreria **LIEF** e comprendono:

- Istogramma ed entropia dei byte
- Statistiche sulle stringhe stampabili
- Attributi dell'header COFF e Optional Header
- Informazioni sulle sezioni, import/export table e data directory

La scelta di queste feature e la loro organizzazione in gruppi segue esattamente la metodologia proposta da OLIVANDER, che ha dimostrato elevata efficacia nella rilevazione di malware per Windows.

> **Nota**: I file CSV utilizzati in questo progetto (`BODMAS_features.csv`) fanno parte del dataset BODMAS, una collezione pubblica di feature statiche di file PE benigni e malevoli.

---

## ⚙️ Requisiti

* Python **3.11 o superiore**
* pip (gestore pacchetti Python)

---

## 🚀 Installazione

### Clonazione del repository

```bash
git clone https://github.com/FrancescoGassi/RagCybersecurity.git
cd RAG_Cybersecurity
```

### Creazione ambiente virtuale (consigliato)

```bash
python -m venv .venv
```

Attivazione:

* **Windows**

```bash
.venv\Scripts\activate
```

* **Linux / macOS**

```bash
source .venv/bin/activate
```

### Installazione dipendenze

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota per Windows:**
> Se `faiss-gpu` non è disponibile, sostituirlo con `faiss-cpu` nel file `requirements.txt`.

---

## 📂 Struttura del Progetto

```
RAG_Cybersecurity/
├── docs/
    ├── DiagrammaClassi.png     # Diagramma UML
├── src/
│   ├── dataset.py              # Gestione dataset (caricamento, MI, split, serializzazione)
│   ├── text_dataset.py         # Dataset testuale (testi + target)
│   ├── embedding_dataset.py    # Dataset embedding (vettori + target)
│   ├── embedding.py            # Generazione embedding (Sentence-Transformer)
│   ├── vector_index.py         # Indice FAISS (costruzione, salvataggio, ricerca)
│   └── feature_names.py        # Nomi simbolici delle 2381 feature
│
├── create_named_csv.py         # Generazione CSV con intestazioni simboliche
├── test_feature_count.py       # Verifica numero di feature
├── test_ml.py                  # Test della classe Dataset
├── test_rag_flow.py            # Pipeline completa RAG
│
├── requirements.txt
└── README.md
```

---

## 📊 Preparazione dei Dati

I file CSV originali devono essere collocati nella directory:

```
DatasetPE/
```

File richiesti:

* `BODMAS_features.csv`
* `test.csv`

### Generazione file con intestazioni simboliche

```bash
python create_named_csv.py
```

Output:

```
*_named.csv
```

---

## 🧪 Esecuzione dei Test

Eseguire i seguenti script nell’ordine indicato.

### 1️. Verifica numero di feature

```bash
python test_feature_count.py
```

**Output atteso:**

* Numero di feature: **2381**
* Visualizzazione delle prime feature

---

### 2️. Test della classe Dataset

```bash
python test_ml.py
```

Verifica:

* Caricamento dei dati
* Calcolo della Mutual Information
* Ordinamento delle feature
* Suddivisione stratificata del dataset
* Serializzazione in formato testuale

---

### 3️. Test completo della pipeline RAG

```bash
python test_rag_flow.py
```

---

## Pipeline RAG

Lo script esegue le seguenti operazioni:

1. Caricamento di un campione di **1000 istanze**
2. Calcolo della Mutual Information
3. Ordinamento delle feature per importanza
4. Suddivisione del dataset:

   * Training: 800
   * Test: 200
5. Conversione in formato testuale:

   ```
   feature: valore
   ```
6. Generazione embedding tramite modello Transformer (`all-MiniLM-L6-v2`)
7. Costruzione indice FAISS (distanza L2)
8. Salvataggio indice e metadati
9. Caricamento dell’indice da disco
10. Ricerca dei **3 vicini più simili**

---

## 📌 Output Atteso
L'esecuzione di test_rag_flow.py produce un output simile al seguente (le distanze e gli indici variano in base al campione):

```
=== Test flusso RAG con FAISS ===

1. Caricamento campione del dataset...
   Caricati 1000 esempi, 2381 feature

2. Calcolo mutual information e ordinamento...
   Top 5 feature: ['section_0', 'import_0', 'byte_entropy_23', ...]

3. Split in train/test...
   Train: 800 esempi, Test: 200 esempi

...

10. Creazione indice FAISS sul training set...
Indice salvato in faiss_index.faiss

11. Caricamento indice da disco...
Indice caricato (dimensione 384)

12. Ricerca top-3 vicini...
   Indici: [760 323 313]
   Distanze: ['12.345678', '15.678901', '18.234567']
   Testi e target dei vicini:
     1: target=1, testo=byte_histogram_0: 0.001234, byte_histogram_1: 0.000567, ... (malware)
     2: target=1, testo=byte_histogram_0: 0.001198, byte_histogram_1: 0.000601, ... (malware)
     3: target=0, testo=byte_histogram_0: 0.003210, byte_histogram_1: 0.002145, ... (goodware)

✅ Flusso RAG completato con successo!
```

---

## 📝 Note Tecniche

* Il test utilizza un sottoinsieme di **1000 campioni** per ridurre i tempi di esecuzione
* File temporanei generati:

  * `train_texts.pkl`
  * `test_texts.pkl`
  * `faiss_index.faiss`
  * `faiss_index_metadata.pkl`

Tali file possono essere eliminati al termine dei test.

---

## Modello di Embedding

* **Modello:** `all-MiniLM-L6-v2`
* **Dimensione embedding:** 384
* **Dimensione modello:** ~80 MB
* **Download:** automatico alla prima esecuzione (Hugging Face)

---
