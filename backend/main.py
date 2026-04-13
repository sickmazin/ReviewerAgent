from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional

from .database import engine, get_db, Base
from .models import Chat, Review
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Inizializzazione Tabelle
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Reviewer Agent API")

# Abilitazione CORS per il Frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAZIONE LLM ---
MODEL_NAME = "gemma3:12b"
llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

class ReviewEvaluationSchema(BaseModel):
    insight_score: int = Field(description="Score 0-100 basato sulla densità informativa.")
    reasoning: str = Field(description="Analisi tecnica del feedback.")

structured_llm = llm.with_structured_output(ReviewEvaluationSchema)

DOMAIN_KNOWLEDGE = {
    "Books": "Focus on narrative structure, character arc consistency, stylistic nuances.",
    "Electronics": "Focus on real-world performance, thermals, battery cycle reliability.",
    "Home": "Focus on structural integrity, material degradation, assembly precision.",
    "Clothing": "Focus on textile density, seam reinforcement, color fastness.",
    "Toys": "Focus on mechanical safety, non-toxicity, impact resistance.",
    "Movies": "Focus on structural pacing, visual composition, audio-visual coherence.",
    "Pet": "Focus on biocompatibility, durability against biting, ease of sanitation.",
    "Sports": "Focus on biomechanical support, thermal regulation, grip coefficient.",
    "Beauty": "Focus on chemical formulation impact (pH), absorption rates, allergens.",
    "Health": "Focus on bioavailability, symptomatic relief, secondary effects.",
    "restaurant": "Focus on food quality, service efficiency, price-to-quantity ratio."
}

# --- SCHEMI PYDANTIC ---
class EvaluateRequest(BaseModel):
    chat_id: int
    text: str
    category: str
    rating: int

class ChatCreate(BaseModel):
    title: str

# --- API ENDPOINTS ---

@app.get("/chats")
def list_chats(db: Session = Depends(get_db)):
    return db.query(Chat).order_by(desc(Chat.created_at)).all()

@app.post("/chats")
def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    db_chat = Chat(title=chat.title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

@app.post("/evaluate")
async def evaluate_review(request: EvaluateRequest, db: Session = Depends(get_db)):
    # 1. Recupero memoria della chat (ultimi 10 punteggi)
    history_scores = db.query(Review.insight_score)\
                       .filter(Review.chat_id == request.chat_id)\
                       .order_by(desc(Review.created_at))\
                       .limit(10).all()
    history_str = ", ".join([str(s[0]) for s in history_scores]) if history_scores else "Nessuno"

    # 2. Preparazione Prompt
    criteria = DOMAIN_KNOWLEDGE.get(request.category, "Focus on facts and actionable pros/cons.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Sei un Analista Senior di Feedback. Valuta l'Insightfulness (0-100) del testo.
            - Usa l'intera scala, evita multipli di 5.
            - Criteri: {criteria}
            - Punteggi recenti in questa chat (evita ripetizioni): {history}
        """),
        ("human", "Rating: {rating}/5\nRecensione: {text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        # 3. Chiamata Ollama
        result = await chain.ainvoke({
            "text": request.text[:1500],
            "criteria": criteria,
            "rating": request.rating,
            "history": history_str
        })
        
        # 4. Salvataggio nel DB
        db_review = Review(
            chat_id=request.chat_id,
            text=request.text,
            rating=request.rating,
            category=request.category,
            insight_score=result.insight_score,
            reasoning={"analysis": result.reasoning}
        )
        db.add(db_review)
        db.commit()
        db.refresh(db_review)
        
        return db_review
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
