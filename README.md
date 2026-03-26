# ReviewerAgent: Hybrid Insightfulness Scoring System

Questo repository implementa una pipeline architetturale avanzata per la valutazione semantica della qualità delle recensioni online, focalizzandosi sul calcolo deterministico dell'**Insightfulness Score**. 

Il sistema è progettato per operare in contesti ad alto volume (es. e-commerce, piattaforme di recensioni) risolvendo il trade-off tra l'alta capacità di ragionamento dei Large Language Models (LLMs) e il costo computazionale proibitivo in fase di inferenza su larga scala.

## Architettura e Approcci Teorici

Il sistema adotta un approccio ibrido che fonde **Knowledge Distillation (Teacher-Student)**, **Multi-Task Learning** e **Unsupervised Heuristics** per quantificare l'utilità intrinseca di un testo.

### 1. Generazione del Dataset e Silver Labeling
Il tracciamento della qualità di una recensione è un problema con label latenti e altamente soggettive. 
Invece di affidarci a metriche proxy rumorose come i "helpful votes", utilizziamo un LLM ad alta capacità come *Teacher* (Zero-Shot Prompting con Structured Output) per annotare un sottoinsieme del dataset Amazon Reviews 2023.

L'LLM valuta la recensione secondo criteri euristici precisi:
*   **Densità Tecnica:** Presenza di misurazioni fisiche, materiali, specifiche.
*   **Contesto d'Uso:** Descrizione empirica dello scenario d'uso.
*   **Analisi Comparativa:** Differenziali rispetto a standard di mercato.
*   **Rilevanza:** Penalizzazione per rumore (es. lamentele logistiche).

Questo genera delle *Silver Labels* continue $\in [0, 100]$ e riduce la varianza delle annotazioni umane.

### 2. Multi-Task Learning (The Student Model)
Il modello di inferenza primario (Student) è basato su `deberta-v3-small`. L'addestramento non si limita a regredire lo score di insightfulness (MSE Loss), ma è formulato come un problema Multi-Task. 

L'encoder di DeBERTa condivide i pesi su più teste (Heads):
*   **Regression Head:** Predice l'Insightfulness Score.
*   **Subjectivity Head:** Classifica il livello di obiettività del testo (BCE Loss).

La funzione di costo congiunta $L = L_{score} + \lambda L_{subj}$ forza la rete ad apprendere rappresentazioni latenti (embedding) più robuste, disentangling l'informazione fattuale dal sentiment puro.

### 3. Hybrid Scoring System
Per mitigare i bias del modello parametrico, l'output neurale è calibrato a runtime (inference phase) con segnali non supervisionati e feature lessicali:
*   **Lexical Density:** Calcolata tramite POS tagging (`spaCy`), misura la concentrazione di entità (PROPN), nomi (NOUN) e aggettivi (ADJ) sul totale dei token. Un'alta densità correla con l'analisi tecnica.
*   **Information Gain (Semantic Entropy):** Misura la Distanza Coseno tra l'embedding della singola recensione e il baricentro (centroide) del manifold della categoria di appartenenza. Recensioni che si discostano dal "rumore di fondo" standard apportano maggiore entropia informativa.

L'equazione di aggregazione finale è una combinazione lineare pesata:
$S_{final} = (S_{model} \cdot (1 - P_{subj})) + (D_{lex} \cdot w_1) + (I_{gain} \cdot w_2) + (V_{helpful} \cdot w_3)$

## Modelli Utilizzati

1.  **`gemma3:12b` (via Ollama):**
    *   **Ruolo:** Teacher LLM per il labeling offline.
    *   **Motivazione:** Elevata capacità di astrazione, eccellente nel seguire schemi strutturati (Pydantic/JSON) e costo nullo in esecuzione locale tramite quantizzazione.
2.  **`microsoft/deberta-v3-small` (44M Params):**
    *   **Ruolo:** Student Encoder per l'inferenza real-time e Multi-task learning.
    *   **Motivazione:** L'architettura Disentangled Attention di DeBERTa V3 si dimostra superiore a BERT/RoBERTa su task di NLU complessi pur mantenendo un footprint compatibile con l'inferenza CPU/Edge.
3.  **`intfloat/multilingual-e5-large` (Riferimento Architetturale RAG):**
    *   **Ruolo:** Embedding model per l'architettura RAG estesa (validazione policy).
    *   **Motivazione:** Top tier per text retrieval, essenziale per estrarre le guideline corrette nello step di classificazione asincrona.
4.  **`en_core_web_sm` (spaCy):**
    *   **Ruolo:** Estrazione feature sintattiche deterministiche (POS Tagging e NER) per il calcolo a runtime della densità lessicale con latenza di frazioni di millisecondo.

## Struttura della Pipeline Sorgente

*   `download_amazon_reviews_5M.py`: Ingestion asincrona dal datalake HuggingFace (Amazon-Reviews-2023) tramite streaming JSONL per limitare l'OOM (Out Of Memory).
*   `generate_labels_ollama.py`: Motore di distillazione. Invia chunk al Teacher LLM forzando la validazione dello schema in Pydantic.
*   `insightful_model.py`: Definizione delle architetture neurali in PyTorch (`InsightScorer`), della logica euristica (`UnsupervisedEngine`) e della classe di orchestrazione (`HybridInsightSystem`).
*   `train_insight_model.py`: Loop di ottimizzazione in PyTorch con AdamW, scheduling lineare e calcolo delle loss differenziate. Supporta il training dinamico sulle Silver Labels.
