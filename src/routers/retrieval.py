from fastapi import APIRouter

from src.schemas.retrieval import QueryRequest
from src.services.retrieval.retrieval_service import RetrievalService

router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"]
)

retrieval_service = RetrievalService()

@router.post("/query")
async def query_documents(
    request:QueryRequest,
):
    results = retrieval_service.search(
        query=request.query
    )
    
    return {
        "query": request.query,
        "results": results
    }