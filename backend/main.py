from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc
import uvicorn
from pydantic import BaseModel, Field
import json
from typing import List, Optional

from database import engine, get_db, Base
import time
from sqlalchemy.exc import OperationalError
from models import Chat, Review
from ml_models.Insightfulness_Model import Insightfulness
from ml_models.rag import ReviewRAGSystem
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def _wait_for_db(engine, timeout: int = 60):
    start = time.time()
    while True:
        try:
            with engine.connect() as conn:
                return
        except OperationalError:
            if time.time() - start > timeout:
                raise
            print("[startup] DB non pronto, riprovo in 1s...")
            time.sleep(1)


# Inizializzazione Tabelle (attendi che Postgres sia pronto)
_wait_for_db(engine, timeout=60)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Reviewer Agent API")

# Abilitazione CORS per il Frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "../.models/v3/best.pt"

# Instantiate Insightfulness scorer once
_INSIGHT_MODEL = Insightfulness(model_path=MODEL_PATH)

# Instantiate a RAG system for sites listing (shared)
_RAG = ReviewRAGSystem()

# --- SCHEMI PYDANTIC ---
class EvaluateRequest(BaseModel):
    chat_id: str
    text: str
    category: str
    rating: int
    model: Optional[str] = "gemma3:27b"


class CreateReviewRequest(BaseModel):
    chat_id: Optional[str] = None
    text: str
    site: str
    url: Optional[str] = None

class ChatCreate(BaseModel):
    title: str

# --- API ENDPOINTS ---

@app.get("/chats")
def list_chats(db: Session = Depends(get_db)):
    rows = db.query(Chat).order_by(desc(Chat.created_at)).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]

@app.post("/chats")
def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    db_chat = Chat(title=chat.title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return {"id": db_chat.id, "title": db_chat.title, "created_at": db_chat.created_at.isoformat()}

@app.post("/evaluate")
async def evaluate_review(request: EvaluateRequest, db: Session = Depends(get_db)):
    try:
        # Process the evaluation using the Insightfulness model
        out = _INSIGHT_MODEL.execute_output(request.text, request.category, llm_model_name=request.model)

        # If chat_id is "0" or empty, create a new Chat first
        effective_chat_id = request.chat_id
        if not effective_chat_id or effective_chat_id == "0":
            chat_title = out.get("title") or (request.text[:40] + "...")
            new_chat = Chat(title=chat_title)
            db.add(new_chat)
            db.commit()
            db.refresh(new_chat)
            effective_chat_id = new_chat.id

        # Create the review record
        db_review = Review(
            chat_id=effective_chat_id,
            text=out.get("review") or request.text,
            site=request.category,
            score=out.get("score"),
            is_generic_compliant=out.get("is_generic_compliant"),
            follow_guidelines=out.get("follow_guidelines"),
            grammar_errors=out.get("grammar_errors"),
            title=out.get("title"),
            reasoning=out.get("reasoning"),
            highlights=out.get("highlights"),
            details=out.get("details"),
        )
        db.add(db_review)
        db.commit()
        db.refresh(db_review)

        return _serialize_review(db_review)
    except Exception as e:
        db.rollback()
        print(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper: serializza una Review SQLAlchemy in dict compatibile frontend
def _serialize_review(r: Review) -> dict:
    # Normalize highlights issues types to frontend expected values
    highlights = r.highlights if r.highlights else None
    if highlights and isinstance(highlights, dict):
        issues = []
        for it in highlights.get("issues", []):
            typ = it.get("type")
            if typ in ("Violazione", "Errore"):
                out_type = "error"
            elif typ in ("Avvertimento", "Da migliorare"):
                out_type = "improve"
            elif typ == "Suggerimento":
                out_type = "suggestion"
            else:
                out_type = "improve"
            issues.append({
                "start": it.get("start"),
                "end": it.get("end"),
                "type": out_type,
                "message": it.get("message"),
                "token": it.get("token"),
                "suggestion": it.get("suggestion"),
                "is_highlight": it.get("is_highlight", True),
            })
        highlights = {"text": highlights.get("text"), "issues": issues}

    # Normalize reasoning to string if possible
    reasoning = r.reasoning
    if isinstance(reasoning, dict):
        # try common keys
        reasoning = reasoning.get("analysis") or reasoning.get("reasoning") or json.dumps(reasoning, ensure_ascii=False)

    return {
        "id": r.id,
        "chat_id": r.chat_id,
        "text": r.text,
        "site": r.site,
        "url": r.url,
        "score": r.score,
        "is_generic_compliant": r.is_generic_compliant,
        "follow_guidelines": r.follow_guidelines,
        "grammar_errors": r.grammar_errors,
        "title": r.title,
        "reasoning": reasoning,
        "highlights": highlights,
        "details": r.details,
        "created_at": r.created_at.isoformat() if r.created_at is not None else None,
    }


@app.get("/sites")
def get_sites():
    # Return platforms from the RAG store as {id, label}
    sites = []
    for pid, data in _RAG.store._store.items():
        sites.append({"id": pid, "label": data.get("display_name", pid)})
    return sites


@app.get("/model-info")
def get_model_info():
    # Minimal model info for the frontend
    return {
        "welcome_title": "Reviewer Agent",
        "welcome_description": "È un sistema full-stack di analisi automatica di recensioni. Dato un testo e una piattaforma di destinazione (Amazon, eBay, ecc.), produce tre output principali: uno score di insightfulness (BAD/GOOD/EXCELLENT), un check di conformità alle linee guida della piattaforma, e un'analisi grammaticale/stilistica con highlight inline sul testo. I risultati vengono salvati in chat persistenti su PostgreSQL.",
        "how_it_works": [
            "Il flusso è parallelo: la recensione viene processata contemporaneamente da due sottosistemi.",
            "L'InsightReviewScorer produce uno score numerico.",
            "Il ReviewRAGSystem recupera le linee guida della piattaforma tramite keyword match (o, in fallback, embedding semantico), poi esegue tre passi tramite LLM locale (Ollama): check euristico, verifica conformità alle linee guida, e rilevamento errori con indici carattere per gli highlight. I due risultati vengono uniti nella risposta finale."
        ],
        "model_details": [
            {"label": "Modello Totale", "value": "Classe Insightfulness che orchestra due componenti"},
            {"label": "InsightReviewScorer", "value": "Rete neurale basata su DeBERTa-v3-small con tre teste: score_head (regressione 0-100), factuality_head (densità informativa), lexical_head (feature linguistiche spaCy). Usa AttentionPooling invece del solo [CLS] per gestire meglio i testi lunghi. Addestrato con una loss composita a 4 componenti (distillazione LLM, fattualità, bound geometrico per categoria, feature lessicali)."},
            {"label": "ReviewRAGSystem", "value": "Sistema RAG con LangChain + Ollama che carica le linee guida da file JSON, le recupera per keyword o embedding, e usa un LLM strutturato (output Pydantic) per i tre check."}
        ],
        "score_categories": [
            {"label": "BAD", "description": "0-36 → Punteggio basso"},
            {"label": "GOOD", "description": "37-70 → Punteggio medio"},
            {"label": "EXCELLENT", "description": "71-100 → Punteggio alto"},
        ],
        "score_dimensions": [],
        "metrics_description": "Il modello produce un valore continuo 0-100 tramite la score_head. Questo score viene poi mappato in tre categorie: 0-36 → BAD, 37-70 → GOOD, 71-100 → EXCELLENT. Il modello impara a dare score alti a recensioni con alta densità lessicale, entità nominali specifiche, struttura argomentativa e assenza di contenuto generico — apprendendo tutto ciò dal testo grezzo senza dipendenze a runtime.",
    }


@app.get("/chats/{chat_id}/review")
def get_chat_reviews(chat_id: str, db: Session = Depends(get_db)):
    rows = db.query(Review).filter(Review.chat_id == chat_id).order_by(desc(Review.created_at)).all()
    return [_serialize_review(r) for r in rows]


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
