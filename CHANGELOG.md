# Changelog

All notable changes to this project are documented here.

The format loosely follows Keep a Changelog
https://keepachangelog.com/en/1.1.0/

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