from fastapi import APIRouter, UploadFile, File, Depends
import shutil
import os
from loguru import logger

from sqlalchemy.orm import Session
from src.core.database import SessionLocal
from src.services.document_service import get_all_documents
from src.schemas.document import DocumentResponse
from src.services.ingestion.ingection_services import IngestionService


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
        

@router.post(
    "/upload", 
    response_model=DocumentResponse
    )
def upload_document(
    file:UploadFile = File(...), 
    db:Session = Depends(get_db)
    ):
    """
    Upload a PDF → extract text → store in DB.
    
    First version of injection pipeline.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file, 
            buffer)
    
    # Call ingestion pipeline
    ingestion_service = (
        IngestionService()
    )
    
    document = (
        ingestion_service.ingect_document(
            db=db,
            file_path=file_path,
            document_name=file.filename
        )
    )
    
    return document

@router.get("/", response_model=list[DocumentResponse])
def list_documents(db:Session = Depends(get_db)):
    """
    Retrive all stored Documents.
    
    To verify the injection of doc in db.
    """
    return get_all_documents(db)