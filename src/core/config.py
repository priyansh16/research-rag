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
    # Where table text comes from (unstructured backend):
    # "ocr" = unstructured's OCR'd cells, flattened (old behaviour);
    # "text_layer" = rebuilt from the PDF text layer, one row per line,
    # falling back to OCR per table when the region has no text layer.
    TABLE_TEXT_SOURCE: str = "ocr"

    # --- Retrieval -----
    MAX_CHUNK_SIZE: int =  1200
    OVERLAP_ELEMENTS:int = 2
    # Cap on the characters carried over as overlap into the next chunk.
    # 0 = no cap (old behaviour: the last OVERLAP_ELEMENTS elements are
    # always carried, however long). When > 0, only trailing elements that
    # fit under the cap are carried, and overlap is dropped if it would push
    # the next chunk over MAX_CHUNK_SIZE.
    CHUNK_OVERLAP_MAX_CHARS: int = 0
    # Split elements longer than MAX_CHUNK_SIZE at sentence boundaries
    # (word boundaries when there are none, e.g. OCR'd tables). False = old
    # behaviour: an oversized element becomes an oversized chunk.
    CHUNK_SPLIT_LONG_ELEMENTS: bool = False
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    TOP_K_RESULTS:int =  3
    # Drop chunks within ±N positions of a higher-ranked chunk from the same
    # document (overlapping chunks are near-duplicates). 0 = off.
    RETRIEVAL_NEIGHBOR_WINDOW: int = 0
    # Cross-encoder that re-orders the vector-search candidates.
    # Adopted in E3: MiniLM beat bge-reranker-base (+9 hit@3, 0 regressions, 3x faster).
    # Empty = reranking off.
    RETRIEVAL_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # How many vector-search candidates the reranker re-scores.
    # 15 vs 20: same quality (hit@3 +1, 0 regressions), p50 latency 301 -> 231 ms.
    RETRIEVAL_RERANK_CANDIDATES: int = 15
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
    