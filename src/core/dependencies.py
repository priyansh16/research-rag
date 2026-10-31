from fastapi import Request, Depends
from loguru import logger

from sqlalchemy.orm import Session
from src.core.database import get_db

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.generation.generation_service import GenerationService
from src.services.generation.ollama_client import OllamaClient
from src.services.guardrails.guardrails_service import GuardrailsService
from src.services.ingestion.ingestion_service import IngestionService
from src.services.document_service import DocumentService

def get_embedding_service(request: Request) -> EmbeddingService:
    logger.debug("Providing shared EmbeddingService.")
    return request.app.state.services.embedding_service


def get_chroma_service(request: Request) -> ChromaService:
    logger.debug("Providing shared ChromaService.")
    return request.app.state.services.chroma_service


def get_retrieval_service(request: Request) -> RetrievalService:
    logger.debug("Providing shared RetrievalService.")
    return request.app.state.services.retrieval_service


def get_guardrails_service(request: Request) -> GuardrailsService:
    logger.debug("Providing shared GuardrailsService.")
    return request.app.state.services.guardrails_service


def get_ollama_client(request: Request) -> OllamaClient:
    logger.debug("Providing shared OllamaClient.")
    return request.app.state.services.ollama_client


def get_generation_service(request: Request) -> GenerationService:
    logger.debug("Providing shared GenerationService.")
    return request.app.state.services.generation_service


def get_ingestion_service(request: Request) -> IngestionService:
    logger.debug("Providing shared IngestionService.")
    return request.app.state.services.ingestion_service

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """
    Constructs DocumentService with the request-scoped DB session.
    """
    return DocumentService(db)
