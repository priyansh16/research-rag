from typing import List,Dict
from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.core.config import settings
from loguru import logger



class RetrievalService:
    """
    Handle semantic retrival using chromaDB.
    """
    
    def __init__(self):
        
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaService()
        
    
    def search(
        self,
        query:str
        ) -> List[Dict]:
        
        # Step 1 - Embed query
        query_embedding = (
            self.embedding_service.embed_text(query)
        )
        
        # Step 2 - Vector search
        results = self.chroma_service.search(
            query_embeddings=query_embedding,
            top_k=settings.TOP_K_RESULTS
        )
        
        # Step 3 - Formate output
        
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
                    "score": 1-distance
                }
            )
        
        return formatted_results
        