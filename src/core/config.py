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
    
    DEBUG: bool = True
    DATABASE_PATH:str = "./rag.db"
    
    PARSER_BACKEND: str = "unstructured"
    
    MAX_CHUNK_SIZE: int =  1200
    OVERLAP_ELEMENTS:int = 2
    
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    TOP_K_RESULTS:int =  3
    
    CHROMA_COLLECTION_NAME: str = "research_documents"
    
    CHROMA_DB_DIR:str = "./chroma_db"
    HEALTH_PROBE_TEXT: str = "health check"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
    