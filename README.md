# RAG Data Infrastructure Platform
![Status](https://img.shields.io/badge/Status-v0.3.0-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)
![Pytest](https://img.shields.io/badge/Pytest-8.0%2B-blue)

> **Retrieval layer: complete and working**  
> **Generation layer: complete and working (local Ollama)**  
> **FastAPI dependency injection: complete**  
> **Application-scoped services: complete**  
> **Evaluation layer: in progress (retrieval harness and experiments)**

---

## Current Release

Current Version: **v0.3.0**

This release refactors the application onto FastAPI dependency
injection and application-scoped service lifecycles. It also adds API-level tests that exercise the real app end-to-end, and splits the test suite into a fast per-commit run and a scheduled Ollama-dependent `e2e` run.

The next milestone (v0.4.0) focuses on adding an evaluation layer and then containerizing the application
with Docker and Docker Compose.

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
- FastAPI dependency injection with application-scoped service
  lifecycle (one shared instance per service, built once at startup)
- Retrieval evaluation harness: hit@k, MRR and answer reachability on a
  96-question golden set, with run-to-run comparison per question
- 105 automated tests covering ingestion, retrieval, reranking, prompt building,
  guardrails, generation, evaluation tooling, and real API/DI wiring
  end-to-end, split into a fast per-commit app suite (45), evaluation
  tooling tests (58, `eval` marker) and a scheduled Ollama-dependent
  `e2e` suite (2)

### In Progress
- Retrieval experiments (chunk dedup, chunking, reranking); see [Retrieval Evaluation](#retrieval-evaluation)
- LangChain integration for prompt management and 
  retrieval-augmented generation
- Alembic migrations for schema versioning
- AWS cloud deployment
- CI workflow wiring up the per-commit / daily `e2e` test split

### Planned
- Streaming response support
- Generation quality evaluation (faithfulness, answer relevance) with RAGAS
- Swap SQLite for PostgreSQL for production readiness

---

## Retrieval Evaluation

A hand-labelled set of 96 questions over the 5 indexed papers (79 answerable, 17 unanswerable),
run directly against `RetrievalService` with no HTTP and no LLM.

| # | Experiment | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | p50 ms | Verdict |
|---|---|---|---|---|---|---|---|---|
| — | Baseline v2 (vector search only) | 37 | **52** | 59 | 66 | 0.583 | 21 | Reference for E2, E3 |
| E0 | Answer reachability | — | — | — | — | — | — | Diagnostic: 78/79 answers reachable |
| E1 | BGE query prefix | | | | | | | Not adopted, no effect (v1 set only) |
| E3 | **Reranker: MiniLM-L-6, 15 candidates** | 55 | **62** | 66 | 70 | 0.756 | 231 | **Adopted** (bge-reranker-base: 6 regressions, 3× slower) |
| E2 | **Reranker + neighbour dedup (window 1)** | 55 | **64** | 67 | 69 | 0.760 | 228 | **Adopted. Current setup** |
| E4 | Chunking: less overlap, split long elements (+ text-layer tables) | 51 | **56** | 60 | 65 | 0.696 | 246 | Not adopted: −6; overlap was adding recall |

Counts are out of 79 answerable questions. hit@3 is the headline metric (the app sends 3 chunks to the LLM).
Vector search alone found the answer in the top 3 for 52/79 questions; with the cross-encoder reranker and
neighbour dedup it's 64/79 (hit@1 37 → 55), at ~210 ms extra per query on CPU.
Results on the earlier 61-question set are kept in EXPERIMENTS.md.
Hypotheses, changes and decisions for each experiment: [EXPERIMENTS.md](evaluation/retrieval/EXPERIMENTS.md).

```bash
# Run the evaluation (writes results/<date>_<label>.json)
uv run python -m evaluation.retrieval.run --label <label>

# Compare two runs question by question
uv run python -m evaluation.retrieval.compare_runs <label_a> <label_b>

# Regenerate the results.md tables from all runs
uv run python -m evaluation.retrieval.summarize_results
```

---

## Roadmap

v0.3.0 (complete)
- FastAPI dependency injection
- Shared service lifecycle
- Singleton embedding model
- API-level tests and e2e/unit test split

v0.4.0
- Evaluations with golden dataset and Recall@ metrics
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

**Application-scoped services via dependency injection**

Every service is constructed exactly once in `main.py`'s `lifespan()`
and stored on `app.state.services` (`src/core/application.py`).
Routers never instantiate a service themselves — they resolve one
through a `Depends(get_*_service)` factory in
`src/core/dependencies.py`. This guarantees a single shared instance
of each service per process (e.g. one loaded embedding model, one
ChromaDB client) and makes dependencies explicit and swappable for
testing, instead of each router/service reaching for a global or
building its own.

## Project Structure

```
research-rag/
├── src/
│   ├── core/
│   │   ├── config.py           # Application configuration using .env
│   │   ├── database.py         # all document store dependencies
│   │   ├── logging.py          # loguru configuration for strctured logging
│   │   ├── middleware.py       # Per-request middleware, bindings logs with unique correlation ID
│   │   ├── application.py      # AppServices: application-scoped service instances
│   │   ├── dependencies.py     # FastAPI Depends factories resolving shared services
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
│   │   │   ├── embedding_service.py        # Embedding generation with retry logic and health probing
│   │   ├── ingestion/
│   │   │   ├── ingestion_service.py        # Full ingestion pipeline.
│   │   ├── parsers/
│   │   │   ├── fitz_parser.py              # Inital extraction pipeline using pdfplumber and fitz
│   │   │   ├── parsing_pipeline.py         # Abstraction layer for fitz, unstructured parser
│   │   │   ├── pdf_utils.py                # Util file used by fitz parser
│   │   │   ├── unstructured_parser.py      # extraction pipeline using unstructured
│   │   ├── retrieval/
│   │   │   ├── retrieval_service.py        # Semantic retrival using chromaDB
│   │   ├── vectorstore/
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
├── evaluation/
│   ├── retrieval/
│   │   ├── run.py                  # Retrieval evaluation harness: hit@k, MRR, reachability per run
│   │   ├── metrics.py              # Pure metric functions (relevance, ranks, hit@k, score separation)
│   │   ├── compare_runs.py         # Question-by-question diff between two runs
│   │   ├── summarize_results.py    # Regenerates results.md from all runs
│   │   ├── questions.jsonl         # Golden question set (answerable + unanswerable)
│   │   ├── EXPERIMENTS.md          # Experiment summary table and decisions
│   │   ├── results.md              # Generated tables for every run (do not edit)
│   │   ├── results/                # One JSON per run (<date>_<label>.json)
│   │   ├── comparisons/            # Saved compare_runs outputs
├── tests/
│   ├── data/                   # sample pdf file
│   ├── conftest.py             #configures different services for whole test suites, like parsing, chunking etc.
│   ├── test_chunking.py        # Test chunking logic
│   ├── test_embeddings.py      # Test embedding logic
│   ├── test_parser.py          # Test unstructured parsing logic
│   ├── test_retrieval.py       # Test retrieval logic
│   ├── test_generation.py       # Test generation logic
│   ├── test_api.py             # API-level tests through FastAPI TestClient (lifespan + DI + routers)
│   ├── test_eval_metrics.py    # Evaluation metrics, question validation, reachability
│   ├── test_compare_runs.py    # Run-to-run comparison
│   ├── test_summarize_results.py # results.md generation
├── pyproject.toml              # Dependency management via uv
├── pytest.ini                  # pytest config, registers the `e2e` marker
├── Makefile                    # `make run` starts the API with reload
├── CHANGELOG.md                # Release notes
├── DEVLOG.md                   # Development journal and learnings
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

# Run all tests (including e2e tests that require a local Ollama server)
uv run pytest

# Run the fast app suite (per-commit, skips e2e and evaluation tooling)
uv run pytest -m "not e2e and not eval"

# Run the evaluation tooling tests (when evaluation/ code changes)
uv run pytest -m eval

# Run only the e2e suite (needs `ollama serve` running locally)
uv run pytest -m "e2e"
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
