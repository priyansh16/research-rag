from sqlalchemy import Column, Integer, String, DateTime
from src.core.database import Base
from datetime import datetime

class Document(Base):
    """
    Represents a stored document in the system.

    This will later be chunked and embedded for RAG.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    
    title = Column(String, nullable=False)
    
    parser = Column(String)
    
    embedding_model = Column(String)
    
    chunk_count = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

