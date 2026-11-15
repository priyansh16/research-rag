"""
Generation service: the RAG "G" step.

Orchestrates the full query -> answer flow:

    1. Guardrails check (reject jailbreak/injection/empty/oversized queries)
    2. Retrieval (reuses RetrievalService.search() — no duplicated logic)
    3. Prompt construction with numbered, citable context
    4. Ollama call to generate a grounded answer
    5. Structured response: answer + resolvable sources + timing

This service does NOT talk to ChromaDB or the embedding model directly —
it delegates to RetrievalService, same as the design decision already
documented in the README (separating retrieval concerns from generation
concerns, mirroring the existing dual-storage separation pattern).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from src.core.config import settings
from src.services.generation.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaGeneration,
)
from src.services.generation.prompt_builder import (
    REFUSAL_MESSAGE,
    build_generation_prompt,
    build_sources_list,
)
from src.services.guardrails.guardrails_service import GuardrailsService
from src.services.retrieval.retrieval_service import RetrievalService


class GenerationBlockedError(Exception):
    """Raised when a query is rejected by guardrails before reaching the LLM."""

    def __init__(self, reason: str, violation_type: str):
        self.reason = reason
        self.violation_type = violation_type
        super().__init__(reason)


@dataclass
class GenerationResult:
    """
    Internal result. The API response is built from the first seven fields
    only; the rest (full context chunks, Ollama counters) are for logging
    and evaluation and never leave the service through the public schema.
    """

    query: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    chunks_used: int = 0
    # Chunks sent to the model, in citation order:
    # {"document_name", "chunk_index", "score", "text"} with the full text.
    context_chunks: List[Dict] = field(default_factory=list)
    # None when Ollama wasn't called (no chunks) or left the field out.
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    ollama: Optional[OllamaGeneration] = None


def to_context_chunks(chunks: List[Dict]) -> List[Dict]:
    """Full-text view of the chunks sent to the model, in citation order."""
    return [
        {
            "document_name": chunk.get("metadata", {}).get("document_name"),
            "chunk_index": chunk.get("metadata", {}).get("chunk_index"),
            "score": chunk.get("score"),
            "text": chunk.get("chunks", ""),
        }
        for chunk in chunks
    ]


class GenerationService:
    """
    Top-level service wiring guardrails + retrieval + Ollama together.

    Each dependency is constructed here directly (matching the existing
    pattern in RetrievalService.__init__ rather than introducing DI at
    this stage) — swap to FastAPI dependency injection later if/when
    services need to be shared/mocked across more routes.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        guardrails_service: GuardrailsService,
        ollama_client: OllamaClient
        ):
        
        self.guardrails_service = guardrails_service
        self.retrieval_service = retrieval_service
        self.ollama_client = ollama_client
        
        logger.info(
            "GenerationService initialized with shared dependencies."
        )

    async def generate(self, query: str) -> GenerationResult:
        total_start = time.perf_counter()

        # Step 1 — Guardrails
        guardrail_result = self.guardrails_service.validate(query)
        if not guardrail_result.is_safe:
            logger.warning(
                "Generation blocked by guardrails: {}", guardrail_result.reason
            )
            raise GenerationBlockedError(
                reason=guardrail_result.reason,
                violation_type=guardrail_result.violation_type,
            )

        # Step 2 — Retrieval (delegates to existing RetrievalService)
        retrieval_start = time.perf_counter()
        chunks = self.retrieval_service.search(query=query)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        if not chunks:
            logger.warning(
                "Generation skipped because retrieval returned no relevant chunks."
                )            
            
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            return GenerationResult(
                query=query,
                answer=REFUSAL_MESSAGE,
                sources=[],
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                generation_latency_ms=0.0,
                total_latency_ms=round(total_latency_ms, 2),
                chunks_used=0,
            )

        # Limit context to top-N chunks even if retrieval returns more —
        # keeps prompt size (and latency/cost) bounded and configurable
        # independently of TOP_K_RESULTS used for retrieval itself.
        context_chunks = chunks[: settings.GENERATION_MAX_CONTEXT_CHUNKS]

        # Step 3 + 4 — Prompt construction + Ollama call
        prompt = build_generation_prompt(query=query, chunks=context_chunks)

        generation_start = time.perf_counter()
        try:
            generation = await self.ollama_client.generate(prompt=prompt)
        except OllamaClientError as exc:
            logger.error("Generation failed: {}", exc)
            raise
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        total_latency_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Generation completed | chunks={} | prompt_tokens={} | output_tokens={} | "
            "retrieval_latency_ms={:.2f} | generation_latency_ms={:.2f} | total_latency_ms={:.2f}",
            len(context_chunks),
            generation.prompt_eval_count,
            generation.eval_count,
            retrieval_latency_ms,
            generation_latency_ms,
            total_latency_ms
        )

        return GenerationResult(
            query=query,
            answer=generation.text,
            sources=build_sources_list(context_chunks),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            chunks_used=len(context_chunks),
            context_chunks=to_context_chunks(context_chunks),
            prompt_eval_count=generation.prompt_eval_count,
            eval_count=generation.eval_count,
            ollama=generation,
        )

    async def health_check(self) -> Dict:
        """Exposed for the observability/health API, mirroring other services."""
        ollama_health = await self.ollama_client.health_check()
        return {
            "healthy": ollama_health.healthy,
            "latency_ms": ollama_health.latency_ms,
            "model": self.ollama_client.model,
            "model_available": ollama_health.model_available,
            "error": ollama_health.error,
        }