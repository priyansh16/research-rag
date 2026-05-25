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

settings = Settings()
    