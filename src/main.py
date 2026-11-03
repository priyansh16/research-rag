from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
import time

from src.core.config import settings
from src.core.database import init_db
from src.core.logging import setup_logging
from src.core.middleware import LoggingMiddleware
from src.routers import health, documents, retrieval, generation

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.retrieval.reranker_service import build_reranker_service
from src.services.guardrails.guardrails_service import GuardrailsService
from src.services.generation.ollama_client import OllamaClient
from src.services.generation.generation_service import GenerationService
from src.services.ingestion.ingestion_service import IngestionService
from src.core.application import AppServices

# Configure logging BEFORE creating the application
setup_logging(debug=settings.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    startup_start = time.perf_counter()
    
    logger.info(f"Starting {settings.APP_NAME}")
    
    init_db()
    
    logger.info("Database initialized")
    
    # Initialize shared application services
    logger.info("Initializing EmbeddingService...")
    embedding_service = EmbeddingService()
    
    logger.info("Initializing ChromaService...")
    chroma_service = ChromaService()
    
    logger.info("Initializing RetrievalService...")
    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
        reranker_service=build_reranker_service(),
    )
    
    logger.info("Initializing GuardrailsService...")
    guardrails_service = GuardrailsService()

    logger.info("Initializing OllamaClient...")
    ollama_client = OllamaClient()
    
    logger.info("Initializing GenerationService...")
    generation_service = GenerationService(
        retrieval_service=retrieval_service,
        guardrails_service=guardrails_service,
        ollama_client=ollama_client,
    )
    
    logger.info("Initializing IngestionService...")
    ingestion_service= IngestionService(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
    )
    
    # Store services on FastAPI application state
    app.state.services = AppServices(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
        retrieval_service=retrieval_service,
        guardrails_service=guardrails_service,
        ollama_client=ollama_client,
        generation_service=generation_service,
        ingestion_service=ingestion_service,
    )
    
    startup_latency = (
        time.perf_counter() - startup_start
    ) * 1000    
    
    logger.success(
        "Application startup completed | startup_latency_ms={:.2f}",
        startup_latency,
    )
    
    yield
    
    logger.info("Shutting down {}", settings.APP_NAME)
    
# App
app = FastAPI(
    title = settings.APP_NAME,
    version = "0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(generation.router)
    
    
@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running"}
