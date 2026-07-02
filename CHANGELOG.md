# Changelog

All notable changes to this project are documented here.

The format loosely follows Keep a Changelog
https://keepachangelog.com/en/1.1.0/

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