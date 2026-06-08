import time
from typing import List 

from loguru import logger 
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from src.core.config import settings


def _is_retryable(exc: BaseException) -> bool:
    """
    Retry on transient runtime/IO errors.
    Do NOT retry on programming errors (TypeError, ValueError, etc.)
    """
    return isinstance(exc, (RuntimeError, OSError, TimeoutError))


class EmbeddingService:
    """
    Handles embedding generation with retry logic and health probing.
    
    Responsibilities:
    - Load embedding model at init
    - Generate embeddings for single texts and batches
    - Expose a health check that verifies the model is live
    - Retry on transient inference errors with exponential backoff
    """
    
    def __init__(self):
        
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        start = time.perf_counter()        
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Embedding model loaded in {elapsed:.0f} ms")
        
    
    # Health probe
    def health(self) -> dict:
        """
        Encode a short probe string to confirm the model is responsive.
        Returns a status dict consumed by the health router.
        """ 
        try:
            start = time.perf_counter()
            self.model.encode(settings.HEALTH_PROBE_TEXT, normalize_embeddings=True)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "status": "ok",
                "model": settings.EMBEDDING_MODEL,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            logger.error(f"Embedding health check failed: {exc}")
            return {
                "status": "degraded",
                "model": settings.EMBEDDING_MODEL,
                "error": str(exc),
            }
        
    #Embedding generation
    @retry(
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )       
    def embed_text(self, text:str) -> List[float]:
        """
        Generate embedding text for single chunk.
        Retries up to 3 times on RuntimeError with exponential backoff (1s → 2s → 4s).
        """
        
        embedding = self.model.encode(text,normalize_embeddings=True)
        return embedding.tolist()
    
    
    @retry(
        retry=retry_if_exception_type(RuntimeError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    ) 
    def embed_texts(self, texts:List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple chunks.
        Retries up to 3 times on RuntimeError with exponential backoff (1s → 2s → 4s).
        """
        embeddings = self.model.encode(texts,normalize_embeddings=True)
        return embeddings.tolist()
        