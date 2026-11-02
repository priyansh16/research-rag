# Changelog

All notable changes to this project are documented here.

The format loosely follows Keep a Changelog
https://keepachangelog.com/en/1.1.0/

---

## [0.4.0] - 2026-11-01

### Added
- Retrieval evaluation harness (`src/services/evaluation/`): 49-question set over 5 papers
  labelled by type (direct / paraphrased / unanswerable) and answer location (body / table / figure);
  reports hit@1/3/5/10, MRR@10 and doc_hit@k overall and per type, location and document;
  compares unanswerable vs relevant scores; records git commit, metric and chunk counts per run.
- Regression test: ingesting two documents stores all chunks for both.
- Test suite grew from 28 to 77 tests.
- `RetrievalService.search()` accepts an optional `top_k` (defaults to `settings.TOP_K_RESULTS`, rejects values < 1), so callers like the evaluation harness can request k per call without mutating global settings.
- Ingest log now reports `submitted` vs actually `stored` chunk counts and warns on partial indexing.


### Changed

- Chroma collection now created with `hnsw:space: cosine` (was default squared L2), so retrieval `score` is true cosine similarity instead of `2·cos − 1`. Rankings are unchanged; applies only to newly created collections.

### Fixed

- `ChromaService.add_document` generated chunk IDs as `chunk_{i}` per document, so every document after the first collided on existing IDs and Chroma silently skipped them — only 97 of 313 chunks were indexed. IDs are now scoped as `{document_id}_chunk_{chunk_index}`.
- Rebuilt existing vector stores (delete `chroma_db/` and `rag.db`, re-upload documents) to recover the missing chunks.

---

## [0.3.0] - 2026-10-31

### Added

- `src/core/application.py`: `AppServices` dataclass holding one application-scoped instance of every service (embedding, chroma, retrieval, guardrails, ollama, generation, ingestion).
- `src/core/dependencies.py`: FastAPI `Depends` factory functions resolving shared instances off `request.app.state.services`.
- `tests/test_api.py`: API-level tests driving the real FastAPI app through `/health`, `/health/full`, `/retrieval/query`, and `/generation/query`, verifying lifespan, DI wiring, and router error-mapping end-to-end.
- `pytest` `e2e` marker (`pytest.ini`) separating the fast per-commit suite from Ollama-dependent tests intended for a scheduled run.
- Updated fixtures in `tests/conftest.py` to `AppServices` changes, wiring with real dependencies the same way application startup does.

### Changed

- `RetrievalService` and `GenerationService` now take their dependencies as constructor arguments instead of instantiating their own sub-services internally
- All routers (`documents`, `generation`, `health`, `retrieval`) resolve services via `Depends(get_*_service)` instead of constructing them inline per request
- `src/main.py` lifespan builds every service exactly once at startup and shares it across requests via `app.state.services`
- Updated `/api/v1/health/full` with `async def` and await the call

### Fixed

- `/api/v1/health/full` with route awaited nothing while calling the async `OllamaClient.health_check()`; route is now `async def` and awaits the call — caught by the new API-level test
- `tests/test_retrieval.py` and `tests/test_generation.py` updated for the new DI constructors; these were broken (4 failing tests) by the dependency-injection refactor before this release

---

## [0.2.0] - 2026-07-02

### Added

- Generation pipeline built on local Ollama
- Prompt builder for retrieval-grounded responses
- Guardrail service protecting against prompt injection and invalid requests
- Structured logging using Loguru
- Request correlation IDs via middleware
- Health probes for generation dependencies
- End-to-end generation tests
- Retry logic for embedding generation

### Changed

- Retrieval pipeline now reused by GenerationService
- Improved service-level logging
- Health endpoint reports component-level readiness
- Better exception handling across services

### Fixed

- Async pytest configuration
- Improved logging consistency
- Better startup diagnostics

---

## [0.1.0]

Initial release.

### Added

- FastAPI project structure
- PDF ingestion
- Semantic chunking
- Unstructured parsing
- ChromaDB vector store
- SQLite metadata store
- Embedding generation
- Semantic retrieval
- Health endpoints