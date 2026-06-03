from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc
import json
import os

from database import get_db
from schema import Chat, Review, ChatCreate, EvaluateRequest

router = APIRouter()

# --- API ENDPOINTS ---

@router.get("/chats")
def list_chats(db: Session = Depends(get_db)):
    """Lists all chats sorted by creation date."""
    rows = db.query(Chat).order_by(desc(Chat.created_at)).all()
    return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]

@router.post("/chats")
def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    """Creates a new chat."""
    db_chat = Chat(title=chat.title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return {"id": db_chat.id, "title": db_chat.title, "created_at": db_chat.created_at.isoformat()}

@router.post("/evaluate")
async def evaluate_review(request: EvaluateRequest, fastapi_request: Request, db: Session = Depends(get_db)):
    """Evaluates a review using the Insightfulness model."""
    try:
        # Get the model from app state
        insight_model = fastapi_request.app.state.insight_model
        
        # Process the evaluation using the Insightfulness model
        out = insight_model.execute_output(request.text, request.category, llm_model_name=request.model)

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


# Helper: serializes a SQLAlchemy Review into a frontend-compatible dict
def _serialize_review(r: Review) -> dict:
    # Normalize highlights issues types to frontend expected values
    highlights = r.highlights if r.highlights else None
    if highlights and isinstance(highlights, dict):
        issues = []
        for it in highlights.get("issues", []):
            typ = it.get("type")
            if typ in ("Violation", "Error"):
                out_type = "error"
            elif typ in ("Warning", "To Improve"):
                out_type = "improve"
            elif typ == "Suggestion":
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


@router.get("/sites")
def get_sites(request: Request):
    """Returns available platforms from the RAG store."""
    rag_system = request.app.state.rag_system
    sites = []
    for pid, data in rag_system.store._store.items():
        sites.append({"id": pid, "label": data.get("display_name", pid)})
    return sites


@router.get("/model-info")
def get_model_info():
    """Returns detailed model information."""
    info_path = os.path.join(os.path.dirname(__file__), ".prompts", "model_info.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model info: {e}")
        return {}


@router.get("/chats/{chat_id}/review")
def get_chat_reviews(chat_id: str, db: Session = Depends(get_db)):
    """Lists all reviews for a given chat."""
    rows = db.query(Review).filter(Review.chat_id == chat_id).order_by(desc(Review.created_at)).all()
    return [_serialize_review(r) for r in rows]


@router.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    """Deletes a chat and its associated reviews."""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"ok": True}
