from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from src.core.config import settings
from src.core.database import init_db
from src.core.logging import setup_logging
from src.core.middleware import LoggingMiddleware
from src.routers import health, documents, retrieval, generation

# Configure logging BEFORE creating the application
setup_logging(debug=settings.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info(f"Starting {settings.APP_NAME}")
    
    init_db()
    
    logger.info("Database initialized")
    
    yield
    
    logger.info("Shutting down {}", settings.APP_NAME)
    
# App
app = FastAPI(
    title = settings.APP_NAME,
    version = "0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(generation.router)
    
    
@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running"}
