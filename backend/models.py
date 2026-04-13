from marshmallow.fields import Boolean
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from .database import Base

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relazione con le recensioni valutate nella chat
    reviews = relationship("Review", back_populates="chat", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    
    # Input
    text = Column(String, nullable=False)
    site = Column(String, nullable=False)
    url = Column(String)
    
    # Output del Modello (Insightfulness Analysis)
    insight_score = Column(Integer,nullable=False)
    guidelines = Column(Boolean,nullable=False)
    guidelines = Column(Boolean,nullable=False)
    gram_errors = Column(Boolean,nullable=False)

    # Reasoning salvato in formato JSONB (Postgres) per massima flessibilità
    reasoning = Column(JSON)
    red_error = Column([(Inter,Inter)]) # Lista di coppia di indici (start, fine) evidenziato nel testo
    yellow_error = Column([(Inter,Inter)]) # Lista di coppia di indici (start, fine) evidenziato nel testo
    suggests = Column([(Inter,Inter)]) # Lista di coppia di indici (start, fine) evidenziato nel testo
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    chat = relationship("Chat", back_populates="reviews")
