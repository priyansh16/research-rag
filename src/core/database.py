from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from loguru import logger

from src.core.config import settings

# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

DATABASE_URL = f"sqlite:///{settings.DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Table initialisation — called once at startup from lifespan
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables defined on Base.metadata.
    Safe to call multiple times — SQLAlchemy skips existing tables.
    """
    logger.info("Initialising database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a Session, always closes it
# get_db lives here so every router imports from one place.
# ---------------------------------------------------------------------------

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()