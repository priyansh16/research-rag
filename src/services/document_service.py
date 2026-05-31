from sqlalchemy.orm import Session
from src.models.document import Document

def create_document(
    db:Session, 
    title:str, 
    parser:str,
    embedding_model: str,
    chunk_count: int,):
    """
    Store a document in the databse.
    """
    doc = Document(
        title=title, 
        parser=parser,
        embedding_model=embedding_model,
        chunk_count=chunk_count,
        )
    
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
    