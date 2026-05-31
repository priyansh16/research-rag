from pydantic import BaseModel
from datetime import datetime


class DocumentResponse(BaseModel):
    id: int
    
    title: str
    
    parser: str
    
    embedding_model: str

    chunk_count: int

    created_at: datetime

    class Config:
        from_attributes = True