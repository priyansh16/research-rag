from loguru import logger

from sqlalchemy.orm import Session


from src.services.parsers.parsing_pipeline import extract_content
from src.services.chunking import create_chunks
from src.services.embeddings.embedding_service import EmbeddingService
from src.services.vectorstore.chroma_service import ChromaService
from src.services.document_service import DocumentService

from src.core.config import settings


class IngestionService:
    """
    Full ingestion pipeline.
    
    Pipeline:
    pdf
    -> parsing
    -> chunking
    -> embeddings
    -> vector storage
    -> metadata storage
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_service: ChromaService
        ):
        self.embedding_service = embedding_service
        
        self.chroma_service = chroma_service
        logger.info(
            "IngestionService initialized with shared dependencies."
        )
    
    def ingect_document(
        self,
        doc_service: DocumentService,
        file_path: str,
        document_name: str
        ):
        
        logger.info(
            "Document ingestion started | filename={}",
            document_name,
        )
        
        # Step 1 - Parse
        text = extract_content(file_path)
        logger.info(
            "Document parsing completed."
        )
        
        # Step 2 - Chunk
        chunks = create_chunks(text)
        
        logger.info(
            "Chunking completed | chunk_count={}",
            len(chunks),
        )
        
        # Step 3 - Embed
        embeddings = (
            self.embedding_service.embed_texts(chunks)
            )
        
        logger.info(
            "Embedding generation completed."
        )
        
        # Step 4 - Store document metadata in SQL
        document = doc_service.create_document(
            title=document_name,
            parser=settings.PARSER_BACKEND,
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_count=len(chunks),
        )
         

        # Step 4 - Build chunk metadata
        metadata = []
        
        for i, _ in enumerate(chunks):
            metadata.append(
                {
                    "document_id": document.id,
                    "document_name": document_name,
                    "chunk_index": i,
                    "parser": settings.PARSER_BACKEND,
                    "embedding_model": settings.EMBEDDING_MODEL,
                }
            )
        
        logger.info(
            "Document metadata stored | document_id={}",
            document.id,
        )
        
        # Step 5 - Store in cromaDb
        self.chroma_service.add_document(
            chunks= chunks,
            embeddings=embeddings,
            metadata=metadata
        )
        
        logger.info(
            "Vector indexing completed."
        )
        
        logger.success(
            "Document ingestion completed successfully | document_id={}",
            document.id,
        )
        
        return document