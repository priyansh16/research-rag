from sqlalchemy import Column, Integer, String, Text
from src.core.database import Base

class Document(Base):
    """
    Represents a stored document in the system.

    This will later be chunked and embedded for RAG.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)