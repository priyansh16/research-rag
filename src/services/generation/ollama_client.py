"""
Thin client for a local Ollama server.

Wraps Ollama's /api/generate endpoint with retry logic and a health
probe, following the same pattern as EmbeddingService — this keeps
the generation dependency observable through the existing health API
instead of being an opaque external call.

Ollama must be running locally: `ollama serve` (often auto-started by
the Mac app/installer). Pull a model first: `ollama pull llama3.2:3b`.
"""

import time
from dataclasses import dataclass

import httpx
from loguru import logger

from src.core.config import settings


class OllamaClientError(Exception):
    """Raised when the Ollama server is unreachable or returns an error."""


@dataclass
class OllamaHealth:
    healthy: bool
    latency_ms: float | None = None
    model_available: bool | None = None
    error: str | None = None


class OllamaClient:
    """
    Minimal async client for Ollama's generate endpoint.

    Uses the non-streaming /api/generate call (stream=False) for
    simplicity — the whole response is returned in one JSON payload.
    Streaming can be added later as a separate method without
    affecting this interface.
    """

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS
        self.max_retries = settings.OLLAMA_MAX_RETRIES

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Retries on connection errors and timeouts with linear backoff.
        Does NOT retry on a 4xx from Ollama (e.g. model not found) —
        that's a config problem, not a transient one, and retrying
        just wastes time before failing anyway.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (
                    temperature
                    if temperature is not None
                    else settings.GENERATION_TEMPERATURE
                ),
                "num_predict": max_tokens or settings.GENERATION_MAX_TOKEN,
            },
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 2):  # +1 initial try
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate", json=payload
                    )

                if response.status_code == 404:
                    raise OllamaClientError(
                        f"Model '{self.model}' not found on Ollama server. "
                        f"Run: ollama pull {self.model}"
                    )
                response.raise_for_status()

                data = response.json()
                return data.get("response", "").strip()

            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning(
                    "Ollama connection failed (attempt {}/{}): {}",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "Ollama request timed out (attempt {}/{}): {}",
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
            except OllamaClientError:
                raise  # config error, don't retry
            except httpx.HTTPStatusError as exc:
                raise OllamaClientError(
                    f"Ollama returned an error: {exc.response.status_code} "
                    f"{exc.response.text}"
                ) from exc

            if attempt <= self.max_retries:
                time.sleep(attempt * 0.5)  # 0.5s, 1s, ... linear backoff

        raise OllamaClientError(
            f"Ollama unreachable at {self.base_url} after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )

    async def health_check(self) -> OllamaHealth:
        """
        Probe the Ollama server and confirm the configured model is pulled.

        Used by the observability/health API — mirrors how ChromaService
        and EmbeddingService report their own component health.
        """
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000

            models = [m["name"] for m in response.json().get("models", [])]
            model_available = any(
                m == self.model or m.startswith(f"{self.model}:") or self.model in m
                for m in models
            )

            if not model_available:
                logger.warning(
                    "Ollama is up but model '{}' is not pulled. Available: {}",
                    self.model,
                    models,
                )

            return OllamaHealth(
                healthy=True,
                latency_ms=round(latency_ms, 2),
                model_available=model_available,
            )
        except Exception as exc:
            logger.error("Ollama health check failed: {}", exc)
            return OllamaHealth(healthy=False, error=str(exc))