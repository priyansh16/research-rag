import time
from fastapi import APIRouter, Depends
from loguru import logger

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.document_service import DocumentService
from src.core.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/v1", tags=["Health"])

# ---------------------------------------------------------------------------
# Dependency helpers — each instantiates the service fresh per request.
# In a larger app we'd inject singletons via lifespan; for now, this is fine .
# ---------------------------------------------------------------------------

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

def get_chroma_service() -> ChromaService:
    return ChromaService()

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
def full_health_check(
    db: Session = Depends(get_db),
):
    """
    Full readiness probe — checks all downstream dependencies.

    Checks:
    - Vector DB (ChromaDB): collection reachable, document count
    - Embedding model: can encode a probe string, reports latency
    - Document store (SQLite): table reachable, document count

    Returns 200 if all dependencies are healthy.
    Returns 503 if any dependency is degraded.
    Use this for readiness probes in Docker/K8s, not liveness.
    """
    start = time.perf_counter()
    logger.info("Running full health check")

    # Run all three checks — collect results even if one fails,
    # so the caller gets a complete picture of what's broken.
    chroma_status = ChromaService().health()
    embedding_status = EmbeddingService().health()

    doc_service = DocumentService(db)
    doc_store_status = doc_service.health()

    total_ms = (time.perf_counter() - start) * 1000

    dependencies = {
        "vector_db": chroma_status,
        "embedding_model": embedding_status,
        "document_store": doc_store_status,
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