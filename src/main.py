from fastapi import FastAPI
from src.routers import health
from loguru import logger

# def lifespan(app: FastAPI):
#     logger.info("Starting RAG API....")
    
app = FastAPI(
    title = "RAG Research Paper Assistant",
    version = "0.1.0"
)

#Include routers
app.include_router(health.router)

@app.on_event("startup")
def startup_event():
    logger.info("Starting RAG API....")
    
@app.get("/")
def root():
    return {"message": "RAG API is running"}
