# Changelog

All notable changes to this project are documented here.

The format loosely follows Keep a Changelog
https://keepachangelog.com/en/1.1.0/

---

## [Unreleased]

### Added
- Retrieval evaluation harness (`evaluation/retrieval/run.py`): 96-question golden set over 5 papers
  (79 answerable, labelled by type direct / paraphrased and answer location body / table / figure;
  17 unanswerable, tagged `near_miss` / `off_topic`), run directly against `RetrievalService`;
  reports hit@1/3/5/10, MRR@10 and doc_hit@3 overall and per type, location and document;
  records git commit, embedding model, distance metric and chunk counts per run.
- Answer reachability check: whether each answer exists in any indexed chunk of its paper
  (the ceiling for every hit@k); misses list the chunks that hold the answer.
- Unanswerable vs relevant score separation, overall and per subtype, to assess "no answer" thresholds.
- `compare_runs.py`: question-by-question diff between two runs (improved, regressed, rank changes,
  config differences, unanswerable score shifts).
- `summarize_results.py`: regenerates `results.md` tables from every run in `results/`.
- `EXPERIMENTS.md`: experiment summary table and decisions (E0 answer reachability; E1 BGE query prefix, not adopted; E2 neighbour dedup, adopted; E3 reranker, adopted;
  E4 chunking overlap fix and text-layer tables, not adopted).
- Evaluation run reports show chunk size stats measured from the stored chunks (mean, p95, max, over the limit).
- Cross-encoder reranking in `RetrievalService.search()` (`RerankerService`, `src/services/retrieval/reranker_service.py`):
  re-scores the top `RETRIEVAL_RERANK_CANDIDATES` (default 15) vector-search chunks with `RETRIEVAL_RERANKER_MODEL`
  (default `cross-encoder/ms-marco-MiniLM-L-6-v2`; empty = off). Results carry the reranker `score` and the original
  `vector_score`. Eval: hit@3 52 → 62 of 79, hit@1 37 → 55; p50 retrieval latency 21 → 231 ms on CPU.
- Neighbour-chunk dropping (`RETRIEVAL_NEIGHBOR_WINDOW`, default 1; 0 = off): skips chunks within ±N positions of a
  higher-ranked chunk from the same document (overlapping chunks are near-duplicates), applied after reranking.
  Eval: +2 hit@3 on top of the reranker (62 → 64 of 79), 0 regressions, no latency cost.
- Evaluation run metadata records `rerank`, `rerank_candidates` and `neighbor_window`.
- `RetrievalService.search()` accepts an optional `top_k` (defaults to `settings.TOP_K_RESULTS`, rejects values < 1), so callers like the evaluation harness can request k per call without mutating global settings.
- Ingest log now reports `submitted` vs actually `stored` chunk counts and warns on partial indexing.
- Test suite grew from 28 to 105 tests: 45 app, 58 evaluation tooling (`eval` marker, run separately), 2 `e2e`.
  Redundant tests were pruned (e.g. `summarize_results` tests, duplicate compare_runs/metrics edge cases) and
  one-service-one-file applied (neighbour-dropping tests live in `test_retrieval.py`).
  Per-commit app run: `pytest -m "not e2e and not eval"`.


### Changed

- Retrieval now reranks by default: the app downloads the MiniLM cross-encoder (~90 MB) on first start, and the
  returned `score` is a cross-encoder logit (unbounded), not cosine similarity. Set `RETRIEVAL_RERANKER_MODEL=""` for
  vector-only retrieval.
- Evaluation `must_contain` matching folds accents (NFKD, combining marks dropped), because OCR'd tables lose diacritics ("Växjö" is stored as "Vaxjo").
- Chroma collection now created with `hnsw:space: cosine` (was default squared L2), so retrieval `score` is true cosine similarity instead of `2·cos − 1`. Rankings are unchanged; applies only to newly created collections.

### Fixed

- Chunker emitted an empty chunk when a document's first element was longer than `MAX_CHUNK_SIZE`, and
  `OVERLAP_ELEMENTS=0` carried the whole previous chunk (`[-0:]` slices everything) instead of none. Neither case
  occurs in the current collection, so stored chunks are unaffected.
- `ChromaService.add_document` generated chunk IDs as `chunk_{i}` per document, so every document after the first collided on existing IDs and Chroma silently skipped them — only 97 of 313 chunks were indexed. IDs are now scoped as `{document_id}_chunk_{chunk_index}`.
- Rebuilt existing vector stores (delete `chroma_db/` and `rag.db`, re-upload documents) to recover the missing chunks.
- `pyproject.toml` version was stale at `0.1.0`; set to the current release, `0.3.0`.

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