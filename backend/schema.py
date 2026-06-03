import uuid
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional

from database import Base

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relation with reviews evaluated in the chat
    reviews = relationship("Review", back_populates="chat", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"))
    
    # Input
    text = Column(String, nullable=False)
    site = Column(String, nullable=False)
    url = Column(String)
    
    # Model Output (Insightfulness Analysis)
    score = Column(String, nullable=True)  # BAD, GOOD, EXCELLENT
    is_generic_compliant = Column(Boolean, nullable=True)
    follow_guidelines = Column(Boolean, nullable=True)
    grammar_errors = Column(Boolean, nullable=True)
    title = Column(String, nullable=True)  # Suggested title by LLM

    # Reasoning and highlights saved in JSON format for maximum flexibility
    reasoning = Column(JSON, nullable=True)  # LLM analysis reasoning
    highlights = Column(JSON, nullable=True)  # Complete structure: text + list of Issues with indices
    details = Column(JSON, nullable=True)  # word_count, char_count

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chat = relationship("Chat", back_populates="reviews")

# --- PYDANTIC SCHEMAS ---
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
