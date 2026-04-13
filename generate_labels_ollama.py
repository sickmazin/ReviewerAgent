import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

# --- CONFIGURAZIONE ---
MODEL_NAME = "gemma3:12b"
INPUT_AMAZON = "datasets/amazon_reviews_5M_random.csv"
INPUT_RESTAURANT = "datasets/Restaurant reviews.csv"
INPUT_BNB = "datasets/airbnb_reviews.csv"  # Nuovo dataset BnB
OUTPUT_CSV = "datasets/reviews_labeled.csv"
SAMPLE_SIZE = 500000

# Mappatura criteri specifici per dominio
DOMAIN_KNOWLEDGE = {
    "Books": "Focus on narrative structure, character arc consistency, stylistic nuances, and thematic execution (avoid spoilers).",
    "Electronics": "Focus on real-world performance (latency, thermals), build tolerances, battery cycle reliability, and UI/UX stability.",
    "Home": "Focus on structural integrity, material degradation over time, ergonomic utility, and assembly precision.",
    "Clothing": "Focus on textile density, seam reinforcement (stitching), fit consistency (true-to-size), and color fastness after washing.",
    "Toys": "Focus on mechanical safety, material non-toxicity, impact resistance (durability), and cognitive engagement levels.",
    "Movies": "Focus on structural pacing, visual composition (cinematography), performative depth, and audio-visual coherence.",
    "Pet": "Focus on biocompatibility (ingredients), durability against biting/scratching, behavioral impact, and ease of sanitation.",
    "Sports": "Focus on biomechanical support, thermal regulation (breathability), grip coefficient, and weight-to-strength ratio.",
    "Beauty": "Focus on chemical formulation impact (pH, allergens), absorption rates, hypoallergenic properties, and cumulative aesthetic results.",
    "Health": "Focus on bioavailability, symptomatic relief precision, secondary effects, and clarity of the administration protocol.",
    "restaurant": "Focus on food quality (organoleptic properties), service efficiency (latency), price-to-quantity ratio, and environmental hygiene.",
    "bnb": "Focus on listing accuracy vs reality, host responsiveness, cleanliness standards, acoustic/thermal comfort, and neighborhood accessibility."
}
DEFAULT_CRITERIA = "Focus on specific usage scenarios, measurable facts, and actionable pros/cons."

class ReviewEvaluation(BaseModel):
    insight_score: int = Field(description="Score 0-100 basato sulla densità informativa specifica per la categoria.")
    reasoning: str = Field(description="Breve analisi dei punti chiave identificati.")

llm = ChatOllama(model=MODEL_NAME, temperature=0.7)
structured_llm = llm.with_structured_output(ReviewEvaluation)

# Buffer per la "memoria" dei punteggi recenti
recent_scores = []

# Prompt Dinamico con Memoria e Rating
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Sei un Analista Senior di Feedback. Valuta la recensione considerando CATEGORIA e CRITERI dell'utente.
        L'Insightfulness Score (0-100) misura quanto la recensione sia utile per una decisione d'acquisto o di soggiorno razionale.
        
        Linee guida per la distribuzione:
        - Usa l'intera scala 0-100 con massima precisione .
        - EVITA assolutamente di restituire sempre gli stessi valori o arrotondare ai multipli di 5.
        - Se vedi che i punteggi recenti sono troppo simili, sforzati di essere più granulare.
        
        Criteri:
        - Low Insight: Menzioni di logistica generica, prezzo senza contesto, o commenti tautologici (es. "Tutto bene").
        - High Insight: Analisi tecnica, scenari d'uso, pro/contro misurabili, dettagli spaziali o comportamentali (host). 
        
        Dominio: {category}
        Criteri Specifici: {criteria}
        Punteggi assegnati recentemente (per coerenza distributiva): {history}
    """),
    ("human", "Recensione da analizzare:\n\n{review_text}")
])

chain = prompt | structured_llm

def get_score_from_llm(text, category, rating):
    global recent_scores
    if not isinstance(text, str) or len(text.strip()) < 20:
        return 15
        
    criteria = DOMAIN_KNOWLEDGE.get(category, DEFAULT_CRITERIA)
    history_str = ", ".join(map(str, recent_scores[-10:])) if recent_scores else "Nessuno"
    
    try:
        result = chain.invoke({
            "review_text": text[:1500], 
            "category": category, 
            "criteria": criteria,
            "rating": rating,
            "history": history_str
        })
        score = min(100, max(0, result.insight_score))
        
        # Aggiorna memoria
        recent_scores.append(score)
        if len(recent_scores) > 50: recent_scores.pop(0)
        
        return score
    except Exception as e:
        print(f"Exception: {e}. Inserisco 50.")


def process_dataset(source="restaurant"):
    """
    Carica il dataset specificato e appende i risultati allo schema unificato.
    source: 'restaurant', 'amazon' o 'bnb'
    """
    # Schema ordinato come l'header del file CSV
    target_cols = ['text', 'helpful_vote', 'rating', 'category', 'insight_score']
    
    if source == "restaurant":
        if not os.path.exists(INPUT_RESTAURANT):
            print(f"File {INPUT_RESTAURANT} non trovato.")
            return
        df = pd.read_csv(INPUT_RESTAURANT, usecols=['text', 'Rating'])
        df = df.rename(columns={'Rating': 'rating'})
        df['category'] = 'restaurant'
        df['helpful_vote'] = 0
    elif source == "bnb":
        if not os.path.exists(INPUT_BNB):
            print(f"File {INPUT_BNB} non trovato.")
            return
        # Header bnb: listing_id,id,date,reviewer_id,reviewer_name,comments
        df = pd.read_csv(INPUT_BNB, usecols=['comments'])
        df.drop(df.head(30000).index,inplace=True)
        df = df.rename(columns={'comments': 'text'})
        df['category'] = 'bnb'
        df['rating'] = 0
        df['helpful_vote'] = 0
    else:
        if not os.path.exists(INPUT_AMAZON):
            print(f"File {INPUT_AMAZON} non trovato.")
            return
        df = pd.read_csv(INPUT_AMAZON, usecols=['text', 'helpful_vote', 'category', 'rating'])

    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df))).copy()
    file_exists = os.path.exists(OUTPUT_CSV)
    
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=f"Processing {source}"):
        category = row['category'] if pd.notnull(row.get('category')) else "general"
        rating = row['rating'] if pd.notnull(row.get('rating')) else 0
        score = get_score_from_llm(row['text'], category, rating)
        
        row_data = {
            'text': row['text'],
            'helpful_vote': row['helpful_vote'] if pd.notnull(row.get('helpful_vote')) else 0,
            'rating': rating,
            'category': category,
            'insight_score': score
        }
        
        # Creazione DataFrame con colonne esplicite e ordinamento coerente
        df_row = pd.DataFrame([row_data], columns=target_cols)
        df_row.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8')
        file_exists = True

if __name__ == "__main__":
    # Esempio di esecuzione per BnB
    process_dataset(source="bnb")
