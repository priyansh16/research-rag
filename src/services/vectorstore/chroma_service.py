import time
from typing import List, Dict

import chromadb
from loguru import logger

from src.core.config import settings



class ChromaService:
    """
    Handles vector database operations.
    
    Responsibilities:
    - Initialize ChromaDB persistent client
    - Store embeddings, documents, and metadata
    - Perform vector similarity search
    - Expose a health check that verifies DB connectivity
    """
    
    def __init__(self):
        logger.info(f"Initializing ChromaDB at: {settings.CHROMA_DB_DIR}")
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME
        )
        logger.info(f"ChromaDB ready - collection '{settings.CHROMA_COLLECTION_NAME}'"
                    f"({self.collection.count()} documents)"
                    )
        
    
    # Health check 
    def health(self):
        """
        Verify the ChromaDB client is reachable and the collection is accessible.
        Returns a status dict consumed by the health router.
        """
        try:
            start = time.perf_counter()
            count = self.collection.count()
            latency_ms = (time.perf_counter() - start) * 1000
            
            return {
                "status": "ok",
                "collection": settings.CHROMA_COLLECTION_NAME,
                "document_count": count,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            logger.error(f"ChromaDB health check failed: {exc}")
            return{
                "status": "degraded",
                "collection": settings.CHROMA_COLLECTION_NAME,
                "error": str(exc),
            }
    
    
    # Write
    def add_document(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ) -> None:
        """
        store chunks + embeddings + metadata.
        """
        
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata
        )
        
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")
        
    # Read
    def search(
        self, 
        query_embeddings: List[float],
        top_k: int
    ) -> Dict:
        """
        Perform vector similarity search.
        """
        start = time.perf_counter()
        
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        logger.debug(
            f"Vector search returned {len(results.get('ids', [[]])[0])} chunks "
            f"in {latency_ms:.1f}ms"
        )
        
        return results
        