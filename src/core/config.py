import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
     Application configuration settings.
    """

    APP_NAME: str = "RAG Assistant"
    
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    
    PARSER_BACKEND: str = os.getenv(
        "PARSER_BACKEND",
        "unstructured"
    )
    
    MAX_CHUNK_SIZE: int = int(
        os.getenv("MAX_CHUNK_SIZE", 1200)
    )

    OVERLAP_ELEMENTS: int = int(
        os.getenv("OVERLAP_ELEMENTS", 2)
    )
    
    EMBEDDING_MODEL: str= os.getenv(
        "EMBEDDING_MODEL", 
        "BAAI/bge-small-en-v1.5"
    )
    
    TOP_K_RESULTS:int = int(os.getenv(
        "TOP_K_RESULTS", 3
    ))
    
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME"
        ,"research_documents")
    
    CHROMA_DB_DIR:str = os.getenv(
        "CHROMA_DB_DIR", 
        "./chroma_db")

settings = Settings()
    