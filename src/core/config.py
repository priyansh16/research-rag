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
    # Drop chunks within ±N positions of a higher-ranked chunk from the same
    # document (overlapping chunks are near-duplicates). 0 = off.
    # On (1) since E4: chunks keep their 2-element overlap (removing it lost
    # 6 questions), and on top of the reranker w1 gave +2 hit@3, 0 regressions.
    RETRIEVAL_NEIGHBOR_WINDOW: int = 1
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
    # Temperature 0 + fixed seed: the same prompt gives the same answer, so
    # evaluation runs are reproducible and grounded answers don't drift.
    GENERATION_TEMPERATURE: float = 0.0
    GENERATION_SEED: int = 42
    GENERATION_MAX_TOKEN: int = 512
    # Context window sent to Ollama. Set explicitly: Ollama truncates prompts
    # longer than its default window without an error, and 5 long chunks plus
    # the instructions can pass 4k tokens.
    OLLAMA_NUM_CTX: int = 8192

    # --- Guardrails ---
    GUARDRAILS_MAX_QUERY_LENGTH: int = 1000
    GUARDRAILS_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
    