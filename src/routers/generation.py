from fastapi import APIRouter, HTTPException, status
from loguru import logger

from src.schemas.generation import GenerationRequest, GenerationResponse
from src.services.generation.generation_service import (
    GenerationBlockedError,
    GenerationService,
)
from src.services.generation.ollama_client import OllamaClientError

router = APIRouter(prefix="/api/v1/generation", tags=["generation"])

generation_service = GenerationService()


@router.post("/query", response_model=GenerationResponse)
async def generate_answer(request: GenerationRequest) -> GenerationResponse:
    """
    Run the full RAG pipeline: validate query -> retrieve chunks ->
    generate a grounded, cited answer via local Ollama model.
    """
    try:
        result = await generation_service.generate(query=request.query)
    except GenerationBlockedError as exc:
        logger.warning("Blocked query rejected at API layer: {}", exc.reason)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "blocked": True,
                "reason": exc.reason,
                "violation_type": exc.violation_type,
            },
        )
    except OllamaClientError as exc:
        logger.error("Ollama unavailable: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Generation service is currently unavailable. "
                "Ensure Ollama is running locally (`ollama serve`)."
            ),
        )

    return GenerationResponse(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        chunks_used=result.chunks_used,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )