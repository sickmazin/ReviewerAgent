# Workflow Architetturale (High-Level)
Il sistema come una Pipeline Sequenziale con RAG, un flusso deterministico che utilizza l'LLM come motore di ragionamento (Chain-of-Thought in un'unica chiamata con output strutturato). Oppure come agenti che lavorano in parallelo alle 3 valutazioni: ortografico, non viola le linee guida, valutazione dell'insightfullness della recensione.
Passaggi logici:
1. Ingestion & Classification:
    Ricezione della recensione e i metadati (es. source: "Amazon").
    Opzionale: Se la fonte non è nota, un piccolo step di classificazione la identifica.
2. Retrieval (RAG):
   Utilizziamo il metadato source per interrogare il Vector Database (o un document store se le policy sono poche).
   Query: "Linee guida per recensioni su {source}".
   Output: Un chunk di testo che contiene le regole specifiche (es. "Su Amazon non citare prezzi di concorrenti").
3. Reasoning Core (L'LLM):
   Costruisci un prompt unico (o a catena) che contiene:
   - Il testo della recensione.
   - Le linee guida recuperate (Context).
   - La definizione di "Insightfulness" .
   - Chiedi all'LLM di produrre un output strutturato (JSON) con le 3 valutazioni.
4. Structured Output Parsing:
   Unificazione dei campi (Ortografia, Compliance, Insightfulness) e output finale.

## Scelta e Gestione dell'LLM
Per questo tipo di task, la scelta del modello dipende dal bilanciamento tra costo, velocità e capacità di ragionamento.
Modelli "Mid-size" molto capaci e veloci. Esempi: Gemini 1.5 Flash, GPT-4o-mini, o Claude 3 Haiku.
La correzione ortografica è banale. Il controllo delle policy richiede attenzione al contesto. L'insightfulness richiede un po' di "ragionamento", ma se la rubrica è chiara, non serve un modello enorme.
Per l'architettura abbiamo bisogno di due tipologie distinte di modelli:
1. Embedding Model (per il RAG): Trasforma le linee guida e le query in vettori numerici.
2. Generative LLM (per il Reasoning): Analizza la recensione e genera il JSON.

Ecco i migliori candidati attuali su Hugging Face, ottimizzati per la lingua italiana e per le performance.
1. Modelli per RAG (Embedding)
   Questi modelli sono specializzati nel capire la similarità semantica. La scelta Top-Tier: `intfloat/multilingual-e5-large` 
   È uno dei migliori modelli multilingua al mondo per il retrieval. Funziona benissimo con l'italiano e capisce la differenza tra "query" (domanda) e "passage" (documento). Dimensioni: Circa 2.2GB (gestibile facilmente).
   L'alternativa leggera: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   Molto veloce e leggerissimo, meno preciso di E5 su sfumature complesse.
2. Modelli LLM (Generazione e Ragionamento)
   Questi modelli devono leggere il contesto recuperato e produrre il JSON. Bastano modelli nella fascia 20B-30B parametri. Si potrebbe optare per `google/gemma-27b`
   Capacità di ragionamento logico sorprendenti ed è ottimo nel seguire istruzioni complesse (come la tua rubrica di insightfulness) e parla un italiano eccellente.
   Oppure `meta-llama/Meta-Llama-3.1-8B-Instruct` poiché estremamente robusto e supporta una context window molto ampia (128k token), utile per molte linee guida. È molto bravo a formattare output in JSON.