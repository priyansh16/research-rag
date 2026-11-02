from typing import List, Dict
import time

from loguru import logger

from src.core.config import settings
from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService



def drop_neighbor_chunks(
    results: List[Dict],
    window: int,
    top_k: int,
) -> List[Dict]:
    """
    Keep results in rank order, skipping any chunk within ±window chunk
    positions of an already-kept chunk from the same document, until
    top_k results are kept. window=0 just truncates to top_k.
    """
    kept: List[Dict] = []
    for result in results:
        if len(kept) == top_k:
            break
        metadata = result["metadata"]
        if window > 0 and any(
            k["metadata"]["document_id"] == metadata["document_id"]
            and abs(k["metadata"]["chunk_index"] - metadata["chunk_index"]) <= window
            for k in kept
        ):
            continue
        kept.append(result)
    return kept


class RetrievalService:
    """
    Handle semantic retrival using chromaDB.
    
    Dependencies are injected during application startup so that
    heavyweight services are shared across the application.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_service: ChromaService,
        ):
        
        self.embedding_service = embedding_service
        self.chroma_service = chroma_service
        
        logger.info(
            "RetrievalService initialized with shared dependencies."
        )
        
    
    def search(
        self,
        query:str,
        top_k: int | None = None,
        ) -> List[Dict]:
        """
        Retrieve the top_k most similar chunks for a query.
        Falls back to settings.TOP_K_RESULTS when top_k is not given.
        """

        k = settings.TOP_K_RESULTS if top_k is None else top_k

        if k < 1:
            raise ValueError("top_k must be >= 1")

        retrieval_start = time.perf_counter()

        logger.info("Retrieval request received.")
        
        # Step 1 - Generate query embedding
        query_embedding = (
            self.embedding_service.embed_text(query)
        )
        
        # Step 2 - Perform vector search. With neighbour dropping on, fetch a
        # larger pool: each kept chunk removes at most 2*window neighbours,
        # so k * (2*window + 1) candidates always leave k results.
        window = settings.RETRIEVAL_NEIGHBOR_WINDOW
        results = self.chroma_service.search(
            query_embeddings=query_embedding,
            top_k=k * (2 * window + 1)
        )
        
        # Step 3 -Format response
        
        formatted_results = []
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for doc, metadata, distance in zip(
            documents, 
            metadatas,
            distances
        ):
            formatted_results.append(
                {
                    "chunks": doc,
                    "metadata": metadata,
                    "score": 1 - distance
                }
            )

        # Step 4 - Drop near-duplicate neighbours, keep top k
        formatted_results = drop_neighbor_chunks(formatted_results, window, k)

        retrieval_latency = (
            time.perf_counter() - retrieval_start
            ) * 1000
        
        if not formatted_results:
            logger.warning(
                "Retrieval completed with no matching documents."
            )
        else:
            logger.info(
                "Retrieval completed | returned_count={} | latency_ms={:.2f}",
                len(formatted_results),
                retrieval_latency,
            )
        
        return formatted_results
        