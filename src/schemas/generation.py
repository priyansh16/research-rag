from typing import List, Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class SourceCitation(BaseModel):
    citation_index: int
    document_name: Optional[str] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    excerpt: str


class GenerationResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    chunks_used: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class GenerationBlockedResponse(BaseModel):
    query: str
    blocked: bool = True
    reason: str
    violation_type: str