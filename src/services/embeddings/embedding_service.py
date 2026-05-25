from sentence_transformers import SentenceTransformer
from typing import List 
from loguru import logger 

from src.core.config import settings

class EmbeddingService:
    """
    Handles embedding generation.
    
    Responsibilities:
    - load embedding model
    - generate embeddings for chunks
    -generate embeddings for queries
    """
    
    def __init__(self):
        
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
    
    def embed_text(self, text:str) -> List[float]:
        """
        Generate embedding text for single chunk.
        """
        
        embedding = self.model.encode(
            text, 
            normalize_embeddings=True)
        
        return embedding.tolist()
    
    def embed_texts(
        self,
        texts:List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple chunks.
        """
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()
        