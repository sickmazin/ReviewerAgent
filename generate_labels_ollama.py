import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

# --- CONFIGURAZIONE ---
MODEL_NAME = "gemma3:12b"
INPUT_CSV = "datasets/amazon_reviews_5M_random.csv"
OUTPUT_CSV = "datasets/amazon_reviews_silver_labels_5M.csv"
SAMPLE_SIZE = 50000

# --- 1. DEFINIZIONE DELLO SCHEMA (Structured Output) ---
# Utilizziamo Pydantic per forzare e validare l'output del modello a livello di libreria
class ReviewEvaluation(BaseModel):
    insight_score: int = Field(description="Un numero intero da 0 a 100 che indica l'utilità e il livello di dettaglio della recensione.")
    reasoning: str = Field(description="Una breve motivazione per il punteggio assegnato.")

# --- 2. INIZIALIZZAZIONE MODELLO ---
# Usiamo ChatOllama che supporta nativamente il binding degli schemi
llm = ChatOllama(model=MODEL_NAME, temperature=0)
structured_llm = llm.with_structured_output(ReviewEvaluation)

# --- 3. DEFINIZIONE DEL PROMPT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Sei un Analista Strategico di Feedback specializzato nella quantificazione della qualità informativa delle recensioni. Il tuo unico compito è assegnare un Insightfulness Score (0-100) basato esclusivamente sul valore intrinseco dei dati forniti nel testo.
        Parametri di Valutazione (Algoritmo Interno):
        Per determinare lo score, analizza la presenza dei seguenti elementi (senza citarli nell'output):
        * Densità Tecnica: Presenza di parametri misurabili, materiali, compatibilità o dettagli costruttivi (Alto impatto).
        * Contesto d'Uso: Descrizione dello scenario di utilizzo, durata del test e problemi specifici riscontrati (Alto impatto).
        * Analisi Comparativa: Confronto con prodotti simili o standard di categoria (Medio impatto).
        * Rilevanza Pura: Una recensione che parla di prezzo, spedizione o imballaggio deve subire una penalizzazione drastica dello score, poiché tali informazioni non riguardano la qualità del prodotto.
        Distribuzione del Punteggio:
        * [0-24] Poor (o): Testi brevi, generici ("ottimo", "non funziona") o focalizzati su logistica/prezzo.
        * [25-49] Fair (X): Descrizione base, opinione soggettiva senza prove concrete o dettagli tecnici.
        * [50-74] Good (A): Buona narrazione dell'esperienza, contesto chiaro, utile per un potenziale acquirente. Fornisce caratteristiche parziali del prodotto.
        * [75-100] Excellent: Analisi da esperto, ricca di dettagli unici, pro e contro bilanciati e specifiche tecniche approfondite. Solo questi punteggi indicano una recensione di alto profilo.
        Formato di Risposta (Rigido):
            Insightfulness:  tra 0 e 100
        Inoltre impara dalle risposte precedenti, e si omogeneo nelle risposte. Non dare sempre lo stesso score anche con testi totalmente differenti.
    """),
    ("human", "Analizza questa recensione e restituisci lo score e il ragionamento:\n\n{review_text}")
])

chain = prompt | structured_llm

def get_score_from_llm(text):
    """Interroga il modello e ottiene l'oggetto Pydantic validato"""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return 50
        
    try:
        # LangChain gestisce internamente la richiesta JSON e il parsing nel modello Pydantic
        result = chain.invoke({"review_text": text[:1500]})
        
        if result and hasattr(result, 'insight_score'):
            return min(100, max(0, result.insight_score))
        return 50
    except Exception as e:
        print(f"\nErrore di validazione o parsing: {e}")
        return 50

def process_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"File {INPUT_CSV} non trovato.")
        return

    print(f"Caricamento dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, usecols=['text', 'helpful_vote', 'rating', 'category'])
    
    # Campionamento
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df))).copy()
    df_sample = df_sample[df_sample['text'].str.len() > 20]
    
    print(f"Inizio elaborazione ({len(df_sample)} recensioni). Salvataggio incrementale attivo...")
    
    # Se il file esiste, lo consideriamo già iniziato, quindi non scriviamo l'header
    file_exists = os.path.exists(OUTPUT_CSV)
    
    # Iterazione riga per riga per salvataggio incrementale
    for index, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        text = row['text']
        score = get_score_from_llm(text)
        
        # Creiamo un dictionary per la riga corrente e aggiungiamo lo score
        row_data = row.to_dict()
        row_data['insight_score'] = score
        
        # Convertiamo in DataFrame per sfruttare il to_csv in append
        df_row = pd.DataFrame([row_data])
        df_row.to_csv(
            OUTPUT_CSV, 
            mode='a', 
            index=False, 
            header=not file_exists, 
            encoding='utf-8'
        )
        file_exists = True # Dopo la prima scrittura, non inserire più l'header

    print(f"\nElaborazione terminata! Dati salvati riga per riga in: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_dataset()
