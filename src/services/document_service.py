import time
from loguru import logger
from sqlalchemy.orm import Session

from src.models.document import Document


class DocumentService:
    """
    Manages all document-level persistence operations.

    Responsibilities:
    - CRUD against the Document table
    - Health check for the document store (SQLite)

    The Session is injected per-request via FastAPI's Depends(get_db).
    DocumentService does not own the session lifecycle — the caller does.
    This keeps transaction boundaries explicit and testable.
    """

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """
        Verify the document store is reachable.
        Runs a lightweight COUNT query — no full table scan.
        """
        try:
            start = time.perf_counter()
            count = self.db.query(Document).count()
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            return {
                "status": "ok",
                "document_count": count,
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            logger.error(f"Document store health check failed: {exc}")
            return {
                "status": "degraded",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create_document(
        self,
        title: str,
        parser: str,
        embedding_model: str,
        chunk_count: int,
    ) -> Document:
        """
        Persist a new document record and return it.
        Commits and refreshes so the returned object has its DB-assigned id.
        """
        doc = Document(
            title=title,
            parser=parser,
            embedding_model=embedding_model,
            chunk_count=chunk_count,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        logger.info(f"Document stored: id={doc.id} title='{doc.title}'")
        return doc

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_documents(self) -> list[Document]:
        """
        Return all stored documents.
        Used for listing, debugging, and evaluation pipelines.
        """
        return self.db.query(Document).all()

    def get_document_by_id(self, document_id: int) -> Document | None:
        """
        Return a single document by primary key, or None if not found.
        Useful for future per-document retrieval and status endpoints.
        """
        return self.db.query(Document).filter(Document.id == document_id).first()