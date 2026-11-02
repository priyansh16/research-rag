from typing import List, Dict
import time

from loguru import logger

from src.core.config import settings
from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService



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
            self.embedding_service.embed_query(query)
        )
        
        # Step 2 - Perform vector search
        results = self.chroma_service.search(
            query_embeddings=query_embedding,
            top_k=k
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
        