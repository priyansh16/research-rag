from typing import List,Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.services.embeddings.embedding_service import EmbeddingService
from src.core.config import settings

class RetrievalService:
    """
    Handles semantic retrival.
    
    Responsibilities:
    - embed query 
    - compare vectors
    - retrieve top-k chunks
    """
    
    def __init__(self):
        self.embedding_services = EmbeddingService()
        
    def search(
        self, 
        query:str, 
        chunk_records: List[Dict]
        ) ->List[Dict]:
        """
        Semantic similarity search
        
        chunk_record_format:
        [
            {
                "chunk": "...",
                "embedding" : [...]
            }
        ]
        """
        
        query_embedding = (
            self.embedding_services.embed_text(query)
            )
        
        similarities = []
        
        for record in chunk_records:
            
            chunk_embedding = record["embedding"]
            
            score = cosine_similarity(
                [query_embedding],
                [chunk_embedding]
            )[0][0]
            
            similarities.append(
                {
                    "chunk": record["chunk"],
                    "score": float(score)
                }
            )
            
        
        similarities.sort(
            key= lambda x :x["score"],
            reverse=True
        )
        
        return similarities[:settings.TOP_K_RESULTS]
        
        