import chromadb
from typing import List, Dict
from loguru import logger
from src.core.config import settings



class ChromaService:
    """
    Handles vector database operations.
    
    Responsibilities:
    - initilize chroma db
    - store embeddings
    - perform vector similarity search
    - manage metadata
    """
    
    def __init__(self):
        
        logger.info("Initializing ChromaDB")
        
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_DIR
        )
        
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME
        )
        
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
        
        
    def search(
        self, 
        query_embeddings: List[float],
        top_k: int
    ) -> Dict:
        """
        Perform vector similarity search.
        """
        
        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k
        )
        
        return results
        