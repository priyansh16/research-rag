from fastapi import APIRouter, UploadFile, File, Depends
import shutil
import os
from loguru import logger

from sqlalchemy.orm import Session
from src.core.database import SessionLocal
from src.services.ingestion.parsing_pipeline import extract_content
from src.services.document_service import create_document, get_all_documents
from src.schemas.document import DocumentResponse
from src.services.ingestion.chunking import create_chunks

router = APIRouter(
    prefix="/api/vi/documents",
    tags=["Documents"]
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

@router.post("/upload", response_model=DocumentResponse)
def upload_document(file:UploadFile = File(...), db:Session = Depends(get_db)):
    """
    Upload a PDF → extract text → store in DB.
    
    First version of injection pipeline.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text
    content = extract_content(file_path)
    
    # Create chunks
    chunks = create_chunks(content)
    logger.info(f"number of chunks: {len(chunks)}")
    logger.info(f"chunck: {chunks}")
    
    # Store in db
    doc = create_document(db, title=file.filename, content=content)
    
    return doc

@router.get("/", response_model=list[DocumentResponse])
def list_documents(db:Session = Depends(get_db)):
    """
    Retrive all stored Documents.
    
    To verify the injection of doc in db.
    """
    return get_all_documents(db)