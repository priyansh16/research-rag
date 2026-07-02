# RAG Data Infrastructure Platform
![Status](https://img.shields.io/badge/Status-v0.2.0-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Pytest](https://img.shields.io/badge/Pytest-8.0%2B-blue)

> **Retrieval layer: complete and working**  
> **Generation layer: complete and working (local Ollama)**  
> **FastAPI dependency injection: in progress**  
> **Evaluation layer: planned**

---

## Current Release

Current Version: **v0.2.0**

This release establishes the complete Retrieval-Augmented Generation
pipeline together with production-oriented observability, request
tracing, structured logging, and automated testing.

The next milestone (v0.3.0) focuses on introducing application-scoped
service lifecycles using FastAPI dependency injection.

---
## Description
An end-to-end document ingestion, retrieval, and observability system built for serving high-quality context to AI agents and LLM pipelines.

## Architecture Overview
The platform follows a FastAPI architecture with dedicated route domains 
for ingestion, retrieval, observability and generation. Each domain is independently 
structured to isolate concerns and enable component-level monitoring. 

### Architecture Diagram
```
                 +----------------------+
                 |       FastAPI        |
                 +----------+-----------+
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
    +---------------+ +---------------+ +---------------+
    |   Ingestion   | |   Retrieval   | |  Generation   |
    +-------+-------+ +-------+-------+ +-------+-------+
            \               |               /
             \              |              /
              +-------------+-------------+
                            |
                            v
                  +-------------------+
                  |     ChromaDB      |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |      SQLite       |
                  +-------------------+
```

The core flows are described below.

1. Ingestion Flow
```txt
PDF Input
    |
Unstructured (text extraction)
    |
Chunking + Embedding Generation
    |
    +----> ChromaDB (vector store)
    |
    +----> SQLite (document store)
```

2. Retrieval Flow
```
Query Input
    |
Embed Query
    |
Vector Search (ChromaDB)
    |
Top-K Chunks
    |
Returned to caller
```

3. Observability Flow
```
Health API
    |
    +----> ChromaDB status
    +----> SQLite (document store) status
    +----> Embedding service status
    --> Structured JSON response
```

4. Generation Flow
```
Query Input
    |
Guardrails
    |
Retrieval flow
    |
Prompt Builder
    |
  Ollama
    |
Grounded Response
```
---

## Tech Stack
| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST API, routing, request validation |
| Vector Store | ChromaDB | Embedding storage and similarity search |
| Document Store | SQLite via SQLAlchemy | Structured document metadata and chunk indexing |
| Embeddings | BAAI/bge-small-en-v1.5 | Sentence-level embedding generation |
| Ingestion | Unstructured | PDF text extraction and preprocessing |
| Observability | Custom Health API | Component-level status and latency monitoring |
| Guardrails | Custom rule based validation | Prevent prompt-injection / jailbreak attempts |
| Generation | Ollama | Local LLM for grounded response generation |
| Language | Python 3.10+ | Core implementation |
| Dependency Management | uv | Fast, modern Python package and project manager |

---
## Current Status

### Working and Tested
- PDF ingestion via Unstructured with configurable chunking
- Embedding generation using BAAI/bge-small-en-v1.5
- Vector storage and similarity search via ChromaDB
- Parallel document metadata indexing in SQLite
- Ranked top-k chunk retrieval with source traceability
- Component-level health monitoring API with per-service 
  latency tracking and structured degradation reporting
- Generation layer via Ollama for local LLM inference with cited documents
- 24 automated unit and integration tests covering ingestion, retrieval, prompt building, guardrails, and generation.

### In Progress
- LangChain integration for prompt management and 
  retrieval-augmented generation
- Alembic migrations for schema versioning
- AWS cloud deployment

### Planned
- Streaming response support
- Retrieval quality evaluation using recall and MRR metrics
- Swap SQLite for PostgreSQL for production readiness

---

## Roadmap

v0.3.0
- FastAPI dependency injection
- Shared service lifecycle
- Singleton embedding model
- Introduction of Evals metrics

v0.4.0
- Docker
- Docker Compose

v0.5.0
- Metrics
- Grafana

v1.0.0
- Production deployment
---
## Design Decisions

**Dual storage pattern**  
ChromaDB handles vector similarity search while SQLite indexes 
structured document metadata separately. Separating these concerns 
means retrieval and storage can evolve independently, and exact 
document lookups do not compete with approximate vector search.

**Component-level observability**  
The health API reports status and latency per dependency rather 
than returning a single binary up/down signal. A degraded document 
store does not mask a healthy vector store, and error messages are 
surfaced directly in the health response for immediate diagnosis 
without log diving.

**Local-first embedding model**  
BAAI/bge-small-en-v1.5 runs fully locally with no external API 
calls, keeping ingestion fast, cost-free, and reproducible across 
environments.

**Request tracing via correlation IDs**  
Each incoming request is tagged with a unique correlation ID at the 
middleware layer and bound to all downstream log entries. This means 
a single failed retrieval request can be traced end-to-end across 
ingestion, embedding, and vector search logs without grepping by 
timestamp. Designed for debuggability from the start, not as an 
afterthought.

**Service composition**
GenerationService reuses RetrievalService rather than querying
ChromaDB directly.

This keeps retrieval logic centralized and avoids duplicated
search implementations. New retrieval strategies automatically
benefit both retrieval and generation endpoints.

**Separation of Retrieval and Generation**

The generation layer never queries the vector database directly.
Instead, GenerationService composes RetrievalService, ensuring a
single implementation of retrieval logic.

This avoids duplicated search implementations and allows retrieval
improvements (ranking, filtering, hybrid search, reranking) to
benefit both APIs automatically.

## Project Structure

```
research-rag/
├── src/
│   ├── core/
│   │   ├── config.py           # Application configuration using .env
│   │   ├── database.py         # all document store dependencies
│   │   ├── logging.py          # loguru configuration for strctured logging
│   │   ├── middleware.py       # Per-request middleware, bindings logs with unique correlation ID
│   ├── routers/
│   │   ├── health.py           # Health and observability endpoints
│   │   ├── documents.py        # Ingestion and document management
│   │   ├── retrieval.py        # Query and retrieval endpoints
│   │   ├── generation.py       # Generation response for query using ollama
│   ├── models/
│   │   ├── document.py        # A stored document model from document store
│   ├── schemas/
│   │   ├── document.py        # Response schema for Ingestion pipeline
│   │   ├── retrieval.py       # Response schema for Retrieval pipeline
│   │   ├── generation.py      # Request and Response schema for Generation pipeline
│   ├── services/
│   │   ├── embeddings/
│   │   │   ├── embeddings_service.py       # Embedding generation with retry logic and health probing
│   │   ├── ingestion/
│   │   │   ├── ingestion_service.py        # Full ingestion pipeline.
│   │   ├── parser/
│   │   │   ├── fitz_parser.py              # Inital extraction pipeline using pdfplumber and fitz
│   │   │   ├── parsing_pipeline.py         # Abstraction layer for fitz, unstructured parser
│   │   │   ├── pdf_utils.py                # Util file used by fitz parser
│   │   │   ├── unstructured_parser.py      # extraction pipeline using unstructured
│   │   ├── retrieval/
│   │   │   ├── retrieval_service.py        # Semantic retrival using chromaDB
│   │   ├── vectorestore/
│   │   │   ├── chroma_service.py           # Handles vector database operations with health probe and logging
│   │   ├── generation/
│   │   │   ├── generation_service.py       # Full Generation pipeline using ollama client
│   │   │   ├── ollama_client.py            # Handles ollama operations with health probe and logging
│   │   │   ├── prompt_builder.py            # Builds the grounded-generation prompt from retrieved chunks
│   │   ├── guardrails/
│   │   │   ├── guardrails_service.py       # Rule-based query guardrails used by generation flow 
│   │   ├── chunking.py         # chunking pipeline
│   │   ├── document_service.py         # Manages all document-level persistence operations with health probe 
│   ├── main.py                  # FastAPI app entry point
├── tests/
│   ├── data/                   # sample pdf file
│   ├── conftest.py             #configures different services for whole test suites, like parsing, chunking etc.
│   ├── test_chunking.py        # Test chunking logic
│   ├── test_embeddings.py      # Test embedding logic
│   ├── test_parser.py          # Test unstructured parsing logic
│   ├── test_retrieval.py       # Test retrieval logic
│   ├── test_generation.py       # Test generation logic
├── pyproject.toml              # Dependency management via uv
└── README.md
```

## Setup and Running

This project uses [uv](https://github.com/astral-sh/uv) for dependency 
management.

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv sync

# Run the API
uv run uvicorn src.main:app --reload

# Run tests
uv run pytest
```
## API Reference

### Health
| Method | Endpoint | Description |
|---|---|---|
| GET | /api/v1/health | Service liveness check |
| GET | /api/v1/health/full | Full dependency health with latency |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/documents/upload/ | Upload PDF, runs full ingestion pipeline |
| GET | /api/v1/documents/ | List all indexed documents |
| GET | /api/v1/documents/{document_id} | Retrieve a specific document |

### Retrieval
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/retrieval/query | Run semantic search, returns ranked top-k chunks |

### Generation
| Method | Endpoint | Description |
|---|---|---|
| POST | /api/v1/generation/query | Run the full RAG pipeline, generate a grounded, cited answer via local Ollama model. |
