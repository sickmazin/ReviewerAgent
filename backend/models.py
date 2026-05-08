import uuid
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from database import Base

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relazione con le recensioni valutate nella chat
    reviews = relationship("Review", back_populates="chat", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"))
    
    # Input
    text = Column(String, nullable=False)
    site = Column(String, nullable=False)
    url = Column(String)
    
    # Output del Modello (Insightfulness Analysis)
    score = Column(String, nullable=True)  # BAD, GOOD, EXCELLENT
    is_generic_compliant = Column(Boolean, nullable=True)
    follow_guidelines = Column(Boolean, nullable=True)
    grammar_errors = Column(Boolean, nullable=True)
    title = Column(String, nullable=True)  # Titolo suggerito dalla LLM

    # Reasoning e highlights salvati in formato JSON per massima flessibilità
    reasoning = Column(JSON, nullable=True)  # Motivazione dell'analisi LLM
    highlights = Column(JSON, nullable=True)  # Struttura completa: text + lista di Issue con indici
    details = Column(JSON, nullable=True)  # word_count, char_count

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chat = relationship("Chat", back_populates="reviews")

