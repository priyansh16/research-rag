from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
import shutil
import os
from loguru import logger

from sqlalchemy.orm import Session
from src.core.database import get_db
from src.services.document_service import DocumentService
from src.schemas.document import DocumentResponse
from src.services.ingestion.ingection_services import IngestionService


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"]
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dependencies 
def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """
    Constructs DocumentService with the request-scoped DB session.
    Injected into route handlers via FastAPI's dependency system.
    """
    return DocumentService(db)

@router.post("/upload", response_model=DocumentResponse)
def upload_document(
    file:UploadFile = File(...), 
    doc_service: DocumentService = Depends(get_document_service),
    db:Session = Depends(get_db)
    ):
    """
    Upload a PDF → run ingestion pipeline → store record in DB.
    
    Pipeline:
      1. Save file to disk
      2. Parse PDF → chunk → embed → store in vector DB (IngestionService)
      3. Persist document metadata to SQLite (DocumentService)
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"Receiving upload: {file.filename}")

    
    # 1. Save to disk 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Run ingestion pipeline
    # IngestionService internally calls DocumentService.create_document
    # We pass db so it can use the same transaction scope.
    try:
        ingestion_service = IngestionService()
        
        document = ingestion_service.ingect_document(
            db=db,
            file_path=file_path,
            document_name=file.filename
        )
    except Exception as exc:
        logger.error(f"Ingestion failed for {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")
    
    return document

@router.get("/", response_model=list[DocumentResponse])
def list_documents( 
    doc_service: DocumentService = Depends(get_document_service),
    ):
    """
    Retrive all stored Documents.
    Used to verify ingestion and for evaluation pipelines.   
    """
    return doc_service.get_all_documents()

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    Return a single document by ID.
    Returns 404 if the document does not exist.
    """
    doc = doc_service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return doc