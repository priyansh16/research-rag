from fastapi import FastAPI
from loguru import logger
from src.core.database import Base, engine

from src.routers import health, documents, retrieval

# def lifespan(app: FastAPI):
#     logger.info("Starting RAG API....")
    
app = FastAPI(
    title = "RAG Research Paper Assistant",
    version = "0.1.0"
)

#Include routers
app.include_router(health.router)

app.include_router(documents.router)

app.include_router(retrieval.router)

@app.on_event("startup")
def startup_event():
    logger.info("Starting RAG API....")
    #create table
    Base.metadata.create_all(bind=engine)
    logger.info("Created table at startup....")
    
    
@app.get("/")
def root():
    return {"message": "RAG API is running"}
