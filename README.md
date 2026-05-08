# 🔍 Reviewer Agent

Un sistema ibrido di analisi automatica di recensioni che combina un modello neurale custom, un sistema RAG e un LLM locale per valutare qualità, conformità e correttezza grammaticale di recensioni su piattaforme come Amazon, eBay, Trustpilot e Google.

---

## Cos'è Reviewer Agent

Reviewer Agent è un'applicazione full-stack che prende in input una recensione testuale e una piattaforma di destinazione, e restituisce un'analisi completa strutturata in tre dimensioni:

- **Insightfulness Score** — quanto è informativa e approfondita la recensione (BAD / GOOD / EXCELLENT)
- **Conformità alle linee guida** — se rispetta le regole della piattaforma (check euristico + LLM)
- **Analisi grammaticale e stilistica** — errori, violazioni e suggerimenti con evidenziazione inline nel testo

Il frontend React visualizza i risultati in modo interattivo, con highlight colorati sul testo originale e un pannello dettagliato per ogni analisi. Le valutazioni sono organizzate in chat persistenti salvate su PostgreSQL.

---

## Come Funziona

Il flusso di elaborazione, una volta inviata una recensione, è il seguente:

```
Input (testo + piattaforma)
        │
        ├─► InsightReviewScorer      → score numerico 0-100 → BAD / GOOD / EXCELLENT
        │
        └─► ReviewRAGSystem
                │
                ├─► GuidelineRetriever   → recupera linee guida della piattaforma
                │       ├─ keyword match (veloce)
                │       └─ semantic match via embedding cosine similarity (fallback)
                │
                ├─► ComplianceChecker
                │       ├─ check euristico (_check_generic)   → violazioni strutturali
                │       ├─ check LLM (check_guidelines_by_model) → conformità contestuale
                │       └─ highlight LLM (check_what_to_highlights) → errori + suggerimenti con indici
                │
                └─► ComplianceResult   → is_generic_compliant, follow_guidelines,
                                         grammar_errors, reasoning, title, highlights, details
```

Il risultato finale unisce lo score di insightfulness con il risultato del RAG checker e viene salvato nel database, poi serializzato e restituito al frontend.

---

## Struttura del Progetto

```
reviewer-agent/
├── backend/
│   ├── main.py                  # FastAPI app, endpoints REST
│   ├── models.py                # Modelli SQLAlchemy (Chat, Review)
│   ├── database.py              # Connessione PostgreSQL
│   └── ml_models/
│       ├── Insightfulness_Model.py   # Facade principale del modello
│       ├── insight_nn_v2.py          # Rete neurale InsightReviewScorer
│       └── rag.py                    # Sistema RAG completo
├── frontend/                    # Applicazione React
├── guidelines/                  # File JSON per ogni piattaforma supportata
│   ├── amazon.json
│   ├── ebay.json
│   └── ...
└── .models/
    └── v3/
        └── best.pt              # Pesi fine-tuned del modello
```

---

## Il Modello Completo

Il modello è composto da due sottosistemi indipendenti che vengono orchestrati dalla classe `Insightfulness`:

### 1. InsightReviewScorer (Rete Neurale)

Basato su **DeBERTa-v3-small** (microsoft/deberta-v3-small), con un'architettura multi-testa:

```
Testo grezzo
     │
     ▼
Tokenizer DeBERTa (max 512 token)
     │
     ▼
Encoder DeBERTa-v3-small  →  hidden states (B, T, H)
     │
     ▼
AttentionPooling           →  vettore (B, H)
     │
     ├─► score_head        →  sigmoid × 100  →  score [0-100]
     ├─► factuality_head   →  sigmoid         →  fattualità [0-1]
     └─► lexical_head      →  sigmoid         →  4 feature lessicali [0-1]
```

#### AttentionPooling

Invece del classico pooling sul solo token `[CLS]`, `AttentionPooling` calcola un'attenzione scalare su ogni token della sequenza e ne fa una media pesata. Questo permette al modello di tenere conto dell'intero testo, distribuendo il peso verso le parti più informative della recensione, indipendentemente dalla loro posizione.

```
scores  = Linear(H → 1) su ogni token        # (B, T)
scores  = masked_fill(padding, -inf)
weights = softmax(scores)                     # (B, T, 1)
pooled  = Σ (hidden_states × weights)         # (B, H)
```

#### Le tre teste

**`score_head`** è la testa principale. Prende il vettore pooled e, attraverso tre layer `Linear → GELU → Dropout` progressivamente più stretti (`H → H/2 → H/4 → 1`), produce un singolo valore scalare. Applicando `sigmoid × 100` si ottiene lo score finale in `[0, 100]`. Questa testa viene addestrata per **distillare il ragionamento di un LLM teacher**, che in fase di data labeling ha assegnato score di insightfulness a ogni recensione.

**`factuality_head`** è una testa ausiliaria che predice la **densità informativa / fattualità** della recensione su scala `[0, 1]`. Il suo target di supervisione è calcolato offline con spaCy come rapporto `(NOUN + PROPN + VERB) / (ADJ + 2)`: un alto numero di sostantivi e verbi rispetto agli aggettivi è proxy di un testo più fattuale e meno vago. Non viene usata a runtime, ma forza l'encoder a sviluppare sensibilità a questo segnale.

**`lexical_head`** è una testa ausiliaria che predice 4 feature lessicali normalizzate, anch'esse calcolate offline con spaCy e usate solo durante il training:

| Feature | Calcolo | Significato |
|---|---|---|
| `noun_ratio` | `(NOUN + PROPN) / tot` | Densità di nomi e entità proprie |
| `verb_ratio` | `VERB / tot` | Densità di verbi d'azione |
| `adj_ratio` | `ADJ / tot` | Densità di aggettivi |
| `entity_density` | `ents / tot` | Presenza di entità named (brand, prodotti, luoghi) |

Questa testa costringe l'encoder a sviluppare una rappresentazione interna sensibile alle proprietà linguistiche del testo, **senza che queste debbano essere calcolate esplicitamente a runtime**.

#### Loss Composita

Il modello viene addestrato con una **loss composita a 4 componenti** (+ 1 opzionale):

| Componente | Peso | Formula | Obiettivo |
|---|---|---|---|
| `L_distill` | α = 0.50 | MSE(score_pred, score_teacher) | Avvicinarsi ai label LLM teacher |
| `L_factuality` | β = 0.15 | MSE(fact_pred, fact_offline) | Sensibilità alla fattualità |
| `L_lexical` | δ = 0.15 | MSE(lex_pred, lex_spacy) | Sensibilità alle proprietà linguistiche |
| `L_bound` | γ = 0.20 | Geometric Bound Loss | Coerenza geometrica per categoria |
| `L_rank` | margin_w = 0.10 | Margin Ranking Loss | Ordinamento relativo corretto |

**Geometric Bound Loss (`L_bound`):** per ogni categoria di prodotto (es. "scarpe", "elettronica") vengono precalcolati due embedding di riferimento: `emb_worst` (la recensione peggiore della categoria) e `emb_best` (la migliore). La loss penalizza il modello se la proiezione dell'embedding corrente sull'asse `worst → best` non rispecchia il suo score target. Questo allinea la geometria dello spazio latente alla qualità effettiva delle recensioni all'interno di ogni categoria.

**Margin Ranking Loss (`L_rank`):** per ogni coppia `(i, j)` nello stesso batch dove `score_i > score_j + 5`, il modello viene penalizzato se predice `rank_i ≤ rank_j`. Garantisce che il modello produca un ordinamento relativo coerente tra recensioni di qualità diversa.

```
L_total = α·L_distill + β·L_factuality + γ·L_bound + δ·L_lexical + margin_w·L_rank
```

#### Training

| Iperparametro | Valore |
|---|---|
| Ottimizzatore | AdamW (lr=2e-7, weight_decay=1e-2) |
| LR Scheduler | CosineAnnealingWarmRestarts (T_0=2, T_mult=2) |
| Gradient accumulation | 20 steps |
| Gradient clipping | max_norm = 1.0 |
| Validation split | 20% |
| Early stopping patience | 15 epoche |
| Metriche di validation | val_loss, MAE, Pearson R |

> **Nota:** le feature spaCy (lexical, factuality) e i category bounds vengono calcolati **una sola volta offline** e cachati su disco. A inferenza il modello lavora esclusivamente sul testo grezzo: nessuna dipendenza da spaCy o da dati di categoria a runtime.

### 2. ReviewRAGSystem (RAG Checker)

Sistema RAG basato su **LangChain + Ollama** (default: `gemma3:27b`) che:

- Carica le linee guida di ogni piattaforma da file JSON in `guidelines/`
- Recupera il documento corretto con match per keyword o embedding semantico
- Esegue tre fasi di analisi via LLM strutturato (Pydantic output):
  - Check euristico (lunghezza, contenuto generico, spam, ecc.)
  - Verifica conformità alle linee guida della piattaforma
  - Rilevamento errori grammaticali e suggerimenti inline con indici carattere

---

## L'Insightfulness Score

L'Insightfulness Score misura **quanto una recensione è informativa, specifica e utile** per altri utenti.

### Come viene calcolato

1. Il testo viene tokenizzato con il tokenizer DeBERTa (max 512 token).
2. L'encoder produce una sequenza di hidden states `(B, T, H)`.
3. `AttentionPooling` computa una rappresentazione pesata sull'intera sequenza.
4. La `score_head` produce un valore continuo da 0 a 100.

### Mappatura in categorie

| Range | Categoria | Significato |
|---|---|---|
| 0 – 36 | **BAD** | Recensione generica, poco informativa |
| 37 – 70 | **GOOD** | Recensione discreta, con contenuto utile |
| 71 – 100 | **EXCELLENT** | Recensione dettagliata, specifica e di alta qualità |

### Cosa impara a riconoscere il modello

Il modello viene addestrato a riconoscere revisioni con:
- Alta densità di sostantivi, verbi d'azione, aggettivi descrittivi specifici
- Presenza di entità nominali (prodotti, brand, caratteristiche tecniche)
- Struttura argomentativa (pro/contro, comparazioni, esperienze personali)
- Assenza di contenuto generico o filler ("ottimo prodotto, consigliato!")

---

## Stack Tecnologico

| Layer | Tecnologia |
|---|---|
| Frontend | React |
| Backend | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy |
| LLM locale | Ollama (`gemma3:27b`) |
| Encoder | DeBERTa-v3-small (HuggingFace Transformers) |
| Embedding RAG | `paraphrase-multilingual-MiniLM-L12-v2` |
| Orchestrazione LLM | LangChain + langchain-ollama |

---

## Setup e Avvio

### Prerequisiti

- Python 3.10+
- Node.js 18+
- PostgreSQL
- [Ollama](https://ollama.com/) con il modello `gemma3:27b` scaricato

```bash
ollama pull gemma3:27b
```

### Backend

```bash
cd backend
pip install -r requirements.txt

# Configura la connessione al DB
export DATABASE_URL="postgresql://user:password@localhost:5432/reviewer_agent"

python main.py
# oppure: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Modello fine-tuned

Posizionare i pesi del modello in `.models/v3/best.pt`. Se il file non è presente, il modello viene inizializzato con pesi casuali (Xavier) e non produrrà score significativi.

---

## API Endpoints

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/chats` | Lista tutte le chat |
| `POST` | `/chats` | Crea una nuova chat |
| `DELETE` | `/chats/{chat_id}` | Elimina una chat |
| `GET` | `/chats/{chat_id}/review` | Recensioni di una chat |
| `POST` | `/evaluate` | Analizza una recensione |
| `GET` | `/sites` | Lista piattaforme disponibili |
| `GET` | `/model-info` | Info sul modello per il frontend |

### Esempio: POST /evaluate

```json
{
  "chat_id": "abc-123",
  "text": "Queste cuffie sono davvero eccellenti. Il suono è cristallino...",
  "category": "amazon",
  "rating": 5,
  "model": "gemma3:27b"
}
```

**Risposta:**

```json
{
  "score": "EXCELLENT",
  "is_generic_compliant": true,
  "follow_guidelines": true,
  "grammar_errors": false,
  "title": "Cuffie eccellenti: audio cristallino e batteria duratura",
  "reasoning": "La recensione è dettagliata, specifica e rispetta tutte le linee guida...",
  "highlights": {
    "text": "Queste cuffie...",
    "issues": []
  },
  "details": { "word_count": 52, "char_count": 312 }
}
```

---

## Aggiungere una Piattaforma

Creare un file `guidelines/<nome_piattaforma>.json` con la seguente struttura:

```json
{
  "display_name": "Nome Piattaforma",
  "keywords": ["nomep", "nome piattaforma"],
  "text": "Testo completo delle linee guida..."
}
```

Il sistema lo caricherà automaticamente al prossimo avvio.