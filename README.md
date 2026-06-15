# 🔍 Reviewer Agent: Sistema Ibrido di Analisi Recensioni

Reviewer Agent è una piattaforma avanzata per l'analisi automatica e intelligente delle recensioni. Combina il meglio di tre mondi: **Fine Tuning (Transformer)** per lo scoring qualitativo, **RAG (Retrieval-Augmented Generation)** per la conformità normativa e **LLM strutturati** per l'analisi grammaticale e stilistica.

Il sistema è progettato per aiutare moderatori e utenti a valutare la qualità informativa (insightfulness), il rispetto delle linee guida di piattaforme specifiche (Amazon, eBay, Google, ecc.) e la correttezza formale del testo.

---

## 🏗️ Architettura del Sistema

Il backend è orchestrato da un'architettura a tre pilastri che elabora ogni recensione in parallelo:

### 1. Il Nucleo Neurale: `InsightReviewScorer`
Un modello basato su **DeBERTaV3-Large** (architettura Transformer all'avanguardia) addestrato con una strategia multi-task per prevedere un punteggio di "informatività".

*   **Attention Pooling**: A differenza dei classici modelli che usano solo il token `[CLS]`, questo sistema implementa un meccanismo di attenzione scalare su ogni singolo token. Questo permette al modello di pesare dinamicamente le parti più rilevanti di una recensione (es. dettagli tecnici vs filler), rendendolo estremamente efficace su testi lunghi.
*   **Architettura Multi-Head**:
    *   **Score Head**: Regressione 0-100 per il punteggio di insightfulness (distillato da un LLM "Teacher", nel caso di sviluppo **Gemma3:27b**).
    *   **Factuality Head**: Predice la densità di informazioni concrete (nomi, verbi) rispetto a aggettivi vaghi.
    *   **Lexical Head**: Apprende internamente le proprietà linguistiche (densità di entità, verbi, nomi) tramite supervisione spaCy in fase di training, senza dipendenze a runtime.
*   **Loss Composita**: Il modello è addestrato minimizzando una loss che combina:
    *   `MSE` per la distillazione dell'LLM.
    *   `Factuality Loss` e `Lexical Loss` per la correzione delle feature spaCy.
    *   `Geometric Bound Loss`: Allinea lo spazio latente degli embedding tra i casi "peggiori" e "migliori" di ogni categoria.
    *   `Margin Ranking Loss`: Garantisce che recensioni migliori abbiano sempre punteggi più alti di quelle peggiori.

### 2. Il Sistema RAG: `ReviewRAGSystem`
Un sistema di recupero intelligente che seleziona dinamicamente le regole della piattaforma corretta.

*   **Guideline Retriever**: Utilizza una strategia a due livelli (Keyword Match + Semantic Embedding) per trovare il documento di policy pertinente alla recensione in un database locale.
*   **Compliance Checker**:
    *   **Check Euristico**: Filtro rapido tramite Regex per identificare violazioni di privacy (telefoni, email), link esterni, spam e uso eccessivo del maiuscolo.
    *   **Check Contestuale**: Utilizza un LLM locale per interpretare se il testo viola le regole della piattaforma.

### 3. Analisi Grammaticale e Highlights
Sempre durante i check precedentemente descritti, un LLM esegue un analisi linguistica che identifica errori oggettivi e suggerisce miglioramenti stilistici.

*   **Highlighting Chirurgico**: Il sistema restituisce indici di carattere precisi (`start`, `end`) per ogni problema, permettendo al frontend di evidenziare il testo in modo interattivo.
*   **Filtro Hallucination**: Un layer di post-processing verifica che gli errori segnalati dall'LLM esistano realmente nel testo originale e corregge eventuali indici errati.

---

## 📊 Analisi dei risultati

Il progetto include un sistema di benchmark (`backend/comparison/`) che confronta il modello locale con i modelli cloud più potenti (Gemma 3.1-Flash-Lite e gpt-oos:120B):
*   **Performance**: Il modello locale `Insightfulness` è ~10x più veloce di Gemini 3.1 Flash (0.2s vs 2.1s per recensione).
*   **Correlazione**: Altissima correlazione (Pearson > 0.85) con i giudizi di modelli LLM top-tier (Gpt e Gemini).
*   **Visualizzazione**: Generazione automatica di grafici di loss, MAE e Pearson R per monitorare il training. 

Inoltre è possibile analizzare i risultati ottenuti in fase di training (`backend/ml_models/plot`) con i seguenti grafici:
*    **Learning Rate per Batch** : Grafico andamento del learning rates per ogni batch.
*    **Loss Components**: Grafico andamento delle componenti principali della loss. 
*    **Validation Metrics**: Grafico delle metriche di validazione (MAE,Pearson).
*    **Loss Curves**: Grafico delle curve di training e validazione.  

Infine l'analisi della composizione del dataset di base (`backend/dataset/plots`) permette di visualizzare diverse caratteristiche quali:
*   **Review Length Dist**: Distribuzione della lunghezza delle recensioni. 
*    **Insight Score Dist**: Distribuzione del punteggio di insightfulness del modello Teacher.
*    **Insight Score by Cat**: Distribuzione del punteggio di insightfulness per ogni categoria.
---

## 🛠️ Stack Tecnologico

*   **Framework**: FastAPI (Python 3.10+)
*   **Database**: PostgreSQL + SQLAlchemy
*   **Machine Learning**: PyTorch, HuggingFace Transformers (DeBERTa-v3)
*   **LLM Engine**: Ollama 
*   **Orchestrazione**: LangChain
*   **NLP**: spaCy 

---

## 🚀 Setup e Installazione

### 1. Requisiti
*   Python 3.10 o superiore.
*   Ollama installato e funzionante.
*   Database PostgreSQL.

### 2. Preparazione Ambiente
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate su Windows
pip install -r requirements.txt
```

### 3. Configurazione Database
Il sistema attende che il DB sia pronto prima di avviarsi. Crea un database chiamato `reviewer_agent` e imposta l'ambiente:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/reviewer_agent"
```

### 4. Modelli e Pesi
Scarica il modello LLM:
```bash
ollama pull gemma3:27b
```
Assicurati che i pesi del modello neurale siano presenti in `backend/.weights/v7_frozen/epoch_117.pt`.

---

## 📁 Struttura del Backend

```
backend/
├── api.py                  # Endpoints REST e logica di business
├── main.py                 # Punto d'ingresso, init DB e caricamento modelli
├── schema.py               # Modelli DB (SQLAlchemy) e Pydantic
├── ml_models/              # Core NLP 
│   ├── Insightfulness_Model.py  # Orchestratore principale (Facade)
│   ├── insightfulness_nn.py     # Architettura Neural Network (PyTorch)
│   └── rag_and_compliance/      # Sistema RAG e logica di controllo
├── dataset/                # Pipeline di data generation e labeling
├── comparison/             # Script per benchmark e comparazioni LLM
└── .prompts/               # Prompt ingegnerizzati per le diverse analisi
```

---

## 🔌 API Endpoints Principali

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `POST` | `/evaluate` | Analisi completa di una recensione (Neural + RAG + LLM) |
| `GET` | `/chats` | Recupera lo storico delle sessioni di analisi |
| `GET` | `/chats/{id}/review` | Dettaglio delle recensioni analizzate in una chat |
| `GET` | `/sites` | Lista delle piattaforme (Amazon, eBay, etc.) supportate |
| `GET` | `/model-info` | Metadati tecnici sul modello per il frontend |

---

## 🧠 Training Pipeline

Se desideri riaddestrare il modello:
1.  **Labeling**: Usa `dataset/generate_labels_ollama.py` per creare "Silver Labels" usando un LLM Teacher.
2.  **Preprocessing**: Il sistema calcola automaticamente le feature spaCy e i category bounds.
3.  **Training**: Esegui `ml_models/training.py` (full fine-tuning) o `ml_models/training_frozen.py` (testa lineare su backbone congelato per risparmio VRAM).

---

*Sviluppato con cura e dedizione da Mattia Corigliano e Paolo Costa.*
