import time
from fastapi import APIRouter, Depends
from loguru import logger

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.document_service import DocumentService
from src.services.generation.ollama_client import OllamaClient
from src.core.dependencies import get_embedding_service, get_chroma_service, get_ollama_client, get_document_service
from src.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1", tags=["Health"])

# Routes
@router.get("/health")
def health_check():
    """
    Lightweight liveness probe.
    Returns 200 immediately — used by load balancers and uptime monitors.
    Does NOT check dependencies (use /health/full for that).
    """
    return {
        "status": "ok",
        "service": "rag-api",
    }


@router.get("/health/full")
async def full_health_check(
    doc_service: DocumentService = Depends(get_document_service),
    chroma_service: ChromaService = Depends(get_chroma_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    ollama_client: OllamaClient = Depends(get_ollama_client)
    ):
    """
    Full readiness probe — checks all downstream dependencies.

    Checks:
    - Vector DB (ChromaDB): collection reachable, document count
    - Embedding model: can encode a probe string, reports latency
    - Document store (SQLite): table reachable, document count
    - Ollama model: model reachable, and its health

    Returns 200 if all dependencies are healthy.
    Returns 503 if any dependency is degraded.
    Use this for readiness probes in Docker/K8s, not liveness.
    """
    start = time.perf_counter()
    logger.info("Running full health check")

    # Run all three checks — collect results even if one fails,
    # so the caller gets a complete picture of what's broken.
    chroma_status = chroma_service.health()
    embedding_status = embedding_service.health()

    doc_store_status = doc_service.health()
    
    ollama_client_status = await ollama_client.health_check()

    total_ms = (time.perf_counter() - start) * 1000

    dependencies = {
        "vector_db": chroma_status,
        "embedding_model": embedding_status,
        "document_store": doc_store_status,
        "ollama_model": ollama_client_status,
    }

    # Overall status is degraded if ANY dependency is degraded
    all_ok = all(v["status"] == "ok" for v in dependencies.values())
    overall_status = "ok" if all_ok else "degraded"

    logger.info(f"Health check complete: {overall_status} ({total_ms:.1f}ms)")

    from fastapi.responses import JSONResponse
    response_body = {
        "status": overall_status,
        "service": "rag-api",
        "total_latency_ms": round(total_ms, 2),
        "dependencies": dependencies,
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(content=response_body, status_code=status_code)