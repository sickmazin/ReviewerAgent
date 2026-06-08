from math import inf
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

# --- CONFIGURATION ---
from backend.dataset.config import MODEL_NAME, INPUT_AMAZON, INPUT_RESTAURANT, INPUT_BNB, OUTPUT_CSV, SAMPLE_SIZE

# Domain-specific criteria mapping
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
    insight_score: int = Field(description="Score 0-100 based on domain-specific information density.")
    reasoning: str = Field(description="Brief analysis of the identified key points.")

llm = ChatOllama(model=MODEL_NAME, temperature=0.6)
structured_llm = llm.with_structured_output(ReviewEvaluation)

# Buffer for "memory" of recent scores
recent_scores = []

# Dynamic Prompt with Memory and Rating
prompt_file_path = os.path.join(os.path.dirname(__file__), "..", ".prompts", "insightfulness_teacher.txt")
with open(prompt_file_path, "r", encoding="utf-8") as f:
    prompt_content = f.read()

prompt = ChatPromptTemplate.from_template(prompt_content)

chain = prompt | structured_llm

def get_score_from_llm(text, category, rating):
    global recent_scores
    if not isinstance(text, str) or len(text.strip()) < 20:
        return 10
        
    criteria = DOMAIN_KNOWLEDGE.get(category, DEFAULT_CRITERIA)
    history_str = ", ".join(map(str, recent_scores[-10:])) if recent_scores else "None"
    
    try:
        result = chain.invoke({
            "review_text": text[:1500], 
            "category": category, 
            "criteria": criteria,
            "rating": rating,
            "history": history_str
        })
        score = min(100, max(0, result.insight_score))
        
        # Update memory
        recent_scores.append(score)
        if len(recent_scores) > 50: recent_scores.pop(0)
        
        return score
    except Exception as e:
        print(f"Exception: {e}. Inserting -inf.")
        return -inf


def process_dataset(source="restaurant"):
    """
    Loads the specified dataset and appends results to the unified schema.
    source: 'restaurant', 'amazon', or 'bnb'
    """
    # Schema ordered as the CSV file header
    target_cols = ['text', 'helpful_vote', 'rating', 'category', 'insight_score']
    
    if source == "restaurant":
        if not os.path.exists(INPUT_RESTAURANT):
            print(f"File {INPUT_RESTAURANT} not found.")
            return

        df = pd.read_csv(INPUT_RESTAURANT, usecols=['text', 'Rating'])
        df = df.rename(columns={'Rating': 'rating'})

        # Add category as it's not present in the original dataset
        df['category'] = 'restaurant'

        # Add column for user votes not present in the original dataset
        df['helpful_vote'] = 0

    elif source == "bnb":
        if not os.path.exists(INPUT_BNB):
            print(f"File {INPUT_BNB} not found.")
            return

        # bnb Header: listing_id,id,date,reviewer_id,reviewer_name,comments
        df = pd.read_csv(INPUT_BNB, usecols=['comments'])
        df.drop(df.head(30000).index, inplace=True)

        # 'comments' is the review text in this dataset
        df = df.rename(columns={'comments': 'text'})

        # Add category as it's not present in the original dataset
        df['category'] = 'bnb'

        # Add columns for missing data in the original dataset
        df['rating'] = 0
        df['helpful_vote'] = 0

    else:
        if not os.path.exists(INPUT_AMAZON):
            print(f"File {INPUT_AMAZON} not found.")
            return

        df = pd.read_csv(INPUT_AMAZON, usecols=['text', 'helpful_vote', 'category', 'rating'])
        
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df))).copy()
    file_exists = os.path.exists(OUTPUT_CSV)
    
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=f"Processing {source}"):
        category = row['category'] if pd.notnull(row.get('category')) else "general"
        rating = row['rating'] if pd.notnull(row.get('rating')) else 0

        # Calling Ollama LLM
        score = get_score_from_llm(row['text'], category, rating)
        
        row_data = {
            'text': row['text'],
            'helpful_vote': row['helpful_vote'] if pd.notnull(row.get('helpful_vote')) else 0,
            'rating': rating,
            'category': category,
            'insight_score': score
        }
        
        # Create DataFrame with explicit columns and consistent ordering
        df_row = pd.DataFrame([row_data], columns=target_cols)
        df_row.to_csv(OUTPUT_CSV, mode='a', index=False, header=not file_exists, encoding='utf-8')
        file_exists = True

if __name__ == "__main__":
    # Example execution for BnB
    process_dataset(source="amazon")
