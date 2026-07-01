import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    Application configuration — validated at startup.
    Reads from environment variables and .env file.
    """
    
    APP_NAME: str = "RAG Assistant"
    
    # --- Ingection -----
    DEBUG: bool = True
    DATABASE_PATH:str = "./rag.db"
    PARSER_BACKEND: str = "unstructured"

    # --- Retrieval -----
    MAX_CHUNK_SIZE: int =  1200
    OVERLAP_ELEMENTS:int = 2
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    TOP_K_RESULTS:int =  3
    CHROMA_COLLECTION_NAME: str = "research_documents"
    CHROMA_DB_DIR:str = "./chroma_db"
    HEALTH_PROBE_TEXT: str = "health check"

    # --- Generation (Ollama) ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_MAX_RETRIES: int = 2
    GENERATION_MAX_CONTEXT_CHUNKS: int = 5      # how many retrieved chunks to feed the LLM
    GENERATION_TEMPERATURE: float = 0.1         # low temp: grounded, less creative drift
    GENERATION_MAX_TOKEN: int = 512

    # --- Guardrails ---
    GUARDRAILS_MAX_QUERY_LENGTH: int = 1000
    GUARDRAILS_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
    