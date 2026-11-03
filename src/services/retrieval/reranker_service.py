import time
from typing import Dict, List, Optional, Sequence

from loguru import logger

from src.core.config import settings


def apply_rerank_scores(results: List[Dict], scores: Sequence[float]) -> List[Dict]:
    """
    Return results re-sorted by reranker score (highest first).

    Each result is copied: `score` becomes the reranker score and the
    original vector similarity is kept as `vector_score`. The sort is
    stable, so equal reranker scores keep their vector-search order.
    """
    if len(results) != len(scores):
        raise ValueError(f"got {len(scores)} scores for {len(results)} results")

    rescored = [
        {**result, "score": float(score), "vector_score": result["score"]}
        for result, score in zip(results, scores)
    ]
    return sorted(rescored, key=lambda r: r["score"], reverse=True)


class RerankerService:
    """
    Re-score (query, chunk) pairs with a cross-encoder.

    The embedding model scores query and chunk separately; a cross-encoder
    reads them together, which ranks better but is too slow to run over the
    whole collection. So it only re-orders a small candidate pool from
    vector search.

    Loaded once at startup, like EmbeddingService. `model` can be injected
    in tests to avoid downloading a real cross-encoder.
    """

    def __init__(self, model_name: str, model=None):
        self.model_name = model_name

        if model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {model_name}")
            start = time.perf_counter()
            model = CrossEncoder(model_name)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Reranker model loaded in {elapsed:.0f} ms")

        self.model = model

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Re-order results by cross-encoder relevance to the query."""
        if not results:
            return []

        start = time.perf_counter()
        pairs = [(query, result["chunks"]) for result in results]
        scores = self.model.predict(pairs)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Reranked candidates | candidates={} | latency_ms={:.2f}",
            len(results),
            latency_ms,
        )
        return apply_rerank_scores(results, list(scores))


def build_reranker_service() -> Optional[RerankerService]:
    """RerankerService for settings.RETRIEVAL_RERANKER_MODEL, or None when reranking is off."""
    if not settings.RETRIEVAL_RERANKER_MODEL:
        return None
    return RerankerService(settings.RETRIEVAL_RERANKER_MODEL)
