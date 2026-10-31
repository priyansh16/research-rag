from dataclasses import dataclass

from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.guardrails.guardrails_service import GuardrailsService
from src.services.generation.generation_service import GenerationService
from src.services.generation.ollama_client import OllamaClient
from src.services.ingestion.ingestion_service import IngestionService

@dataclass
class AppServices:
    """
    Holds application-scoped service instances.

    These services are created once during application startup
    and shared across all incoming requests.
    """
    
    embedding_service: EmbeddingService
    chroma_service: ChromaService
    retrieval_service: RetrievalService
    guardrails_service: GuardrailsService
    ollama_client: OllamaClient
    generation_service: GenerationService
    ingestion_service: IngestionService