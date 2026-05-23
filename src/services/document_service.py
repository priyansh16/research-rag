from sqlalchemy.orm import Session
from src.models.document import Document

def create_document(db:Session, title:str, content:str):
    """
    Store a document in the databse.
    First step in building a knowledge base.
    """
    doc = Document(title=title, content=content)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

def get_all_documents(db:Session):
    """
    Fetch all stored documents.
    Will be used in later evalution and debugging.
    """
    return db.query(Document).all()
    