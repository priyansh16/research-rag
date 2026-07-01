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
from src.services.generation.ollama_client import OllamaClient, OllamaClientError
from src.services.generation.prompt_builder import (
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
    query: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    chunks_used: int = 0


class GenerationService:
    """
    Top-level service wiring guardrails + retrieval + Ollama together.

    Each dependency is constructed here directly (matching the existing
    pattern in RetrievalService.__init__ rather than introducing DI at
    this stage) — swap to FastAPI dependency injection later if/when
    services need to be shared/mocked across more routes.
    """

    def __init__(self) -> None:
        self.guardrails_service = GuardrailsService()
        self.retrieval_service = RetrievalService()
        self.ollama_client = OllamaClient()

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
            logger.info("No chunks retrieved for query; returning early.")
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            return GenerationResult(
                query=query,
                answer=(
                    "I couldn't find any relevant information in the indexed "
                    "documents to answer this question."
                ),
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
            answer = await self.ollama_client.generate(prompt=prompt)
        except OllamaClientError as exc:
            logger.error("Generation failed: {}", exc)
            raise
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        total_latency_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Generation complete: retrieval={}ms generation={}ms total={}ms chunks={}",
            round(retrieval_latency_ms, 2),
            round(generation_latency_ms, 2),
            round(total_latency_ms, 2),
            len(context_chunks),
        )

        return GenerationResult(
            query=query,
            answer=answer,
            sources=build_sources_list(context_chunks),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            chunks_used=len(context_chunks),
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