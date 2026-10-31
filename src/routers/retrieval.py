from fastapi import APIRouter, Depends

from src.schemas.retrieval import QueryRequest
from src.services.retrieval.retrieval_service import RetrievalService
from src.core.dependencies import get_retrieval_service

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"]
)

@router.post("/query")
async def query_documents(
    request:QueryRequest,
    retrieval_service: RetrievalService = Depends(
        get_retrieval_service
    ),
):
    results = retrieval_service.search(
        query=request.query
    )
    
    return {
        "query": request.query,
        "results": results
    }