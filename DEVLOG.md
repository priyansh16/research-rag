# Developer Log

## Setup

### Implementation
- Created clean project structure for scalable RAG development
- Initialized environment using UV package manager
- Configured virtual environment and dependency management
- Built initial FastAPI application
- Added health-check endpoint for service validation
- Verified local API execution and routing

### Learning
- Early project structure matters significantly for maintainability
- Keeping services modular from the start simplifies future scaling
- UV provides faster and cleaner dependency management compared to traditional pip workflows

---

## PDF Extraction Pipeline

### Implementation
- Built PDF ingestion pipeline
- Implemented text extraction using PyMuPDF (`fitz`)
- Implemented table extraction using `pdfplumber`
- Converted extracted tables into RAG-friendly natural language
- Built modular extraction architecture:
  - `extract_content()` → abstraction layer
  - `process_pdf()` → orchestration layer
  - `pdf_utils.py` → extraction and cleaning helpers
- Integrated extraction pipeline with FastAPI upload endpoint
- Stored extracted content into database
- Added cleaning pipeline:
  - hyphenation fixes
  - whitespace normalization
  - noise cleanup

### Learning

#### PDFs Are Structurally Fragile
PDFs do not store semantic structure directly.
Most PDFs internally store positional drawing instructions rather than paragraphs, sections, or tables.

This makes extraction highly dependent on layout interpretation.


#### Layout Matters More Than Raw Text
Good extraction is not simply about retrieving text.
Preserving:
- sections
- headings
- reading order
- tables
- semantic hierarchy

is critical for downstream retrieval quality.


#### Table Extraction Is Non-Trivial
Table extraction using heuristic tools like `pdfplumber` is inconsistent for real-world PDFs.

Challenges observed:
- visually aligned tables not detected structurally
- multi-column layouts breaking table boundaries
- table content merging into narrative text
- inconsistent row detection


#### Converting Tables Into Natural Language Helps Retrieval
Transforming tables into sentence-like representations improves compatibility with retrieval pipelines and embeddings.

Example:
```text
Situation is Reducing deficit, Average GPIV is 0.249, Goals is 887.
```
This preserves semantic meaning better than raw tabular formatting.


#### Separation of Concerns Is Critical

Keeping:
- extraction
- cleaning
- orchestration
- chunking

isolated into separate modules significantly improves maintainability and debugging.


#### Limitations
- Table extraction remains unreliable for complex layouts
- Multi-column PDFs may break reading order
- Structural metadata is mostly lost during extraction
- No semantic document understanding yet
- Extraction quality directly impacts downstream chunking quality

---

## Initial Chunking Pipeline

### Implementation
- Built recursive chunking pipeline
- Added paragraph splitting
- Added sentence splitting
- Added chunk overlap support
- Added basic table-aware chunk separation
- Implemented configurable chunk size and overlap parameters

### Learning
1. Chunking Is Not Simple Text Splitting.

Chunking quality depends heavily on preserving semantic structure.

Poor extraction leads to:

- broken semantic boundaries
- diluted embeddings
- low-quality retrieval

2. Never Flatten Structure Before Chunking

Flattening line breaks and paragraphs before chunking destroys semantic boundaries.

Bad:
- converting all newlines into spaces before chunking

Better:
- preserve structure until chunking is complete

3. Noisy Chunks Are Better Than Missing Information

Aggressive cleaning heuristics can accidentally remove critical information.

Examples:
- dates
- metrics
- identifiers
- statistics
- individual's data

RAG systems generally prefer slightly noisy retrieval over incomplete retrieval.

4. Heuristic Cleaning Does Not Scale

Simple heuristics such as:
- digit counting
- regex-only table detection
- line filtering

break easily across different document types.

A heuristic that works for one PDF often fails for another.

5. Chunking Depends On Extraction Quality
Chunking cannot fully compensate for poor extraction.

If semantic structure is lost during extraction:
- headings merge into paragraphs
- sections collapse
- pages become single giant chunks
- embeddings lose specificity

6. Hierarchical Chunking Is Better Than Fixed Splitting

Chunking should preserve:
- sections
- semantic groups
- headings
- layout hierarchy

Recursive and semantic chunking strategies are significantly more robust than naive fixed-size splitting.

## Limitations
- Chunking still depends on weak extraction structure
- Heading detection is heuristic-based
- Structural metadata is missing
- Tables are not semantically preserved
- Chunking quality varies heavily across document types

## Architectural Decisions
The project is evolving from:
```
PDF → plain text → chunking
```
toward:
```text
PDF → semantic document elements → intelligent chunking
```


## Planned Improvements
- Replace low-level extraction pipeline with Unstructured
- Preserve semantic document elements
- Build metadata-aware chunking
- Add embedding generation
- Implement vector retrieval
- Add hybrid retrieval (BM25 + vector search)
- Add evaluation pipeline using RAGAS and retrieval metrics

## Key Engineering Insights
- RAG quality depends heavily on document structure preservation
- Extraction quality directly affects retrieval quality
- Chunking is fundamentally a document understanding problem
- Semantic preservation is more important than aggressive cleaning
- Retrieval systems prefer noisy truth over missing truth
- PDFs are significantly harder to process than plain text documents

---

## Unstructured for document parcing
- extraction quality defines RAG quality
- semantic parsing > regex cleanup
- unstructured preserves layout-aware document elements
- hi_res parsing uses OCR/layout analysis
- tables require document-aware parsing, not heuristics
- flattening documents destroys semantic structure
- semantic chunking depends on semantic extraction
- production RAG starts with document intelligence
- dependency isolation exposed hidden package coupling
- pyproject.toml + uv.lock improves reproducibility

- semantic extraction enables semantic chunking
- section-aware chunks improve retrieval quality
- chunking should preserve meaning, not token counts
- hi_res parsing improves layout understanding
- metadata-aware chunks are critical for production RAG
- document structure is retrieval signal

# Final chunking Pipeline
- Never overlap with hard codded characters, it may break sentances, or words in mid way. 
- overlap corruption

---

## Dependency Injection & Application-Scoped Services (v0.3.0)

### Implementation
- `AppServices` dataclass holds one shared instance of every service
- Services built once at startup in `main.py` lifespan, shared via `app.state`
- `Depends(get_*_service)` factories in `dependencies.py`, used by all routers
- `RetrievalService` / `GenerationService` now take dependencies in `__init__` instead of building their own
- Added `tests/test_api.py` — real API calls via `TestClient`, not just service logic
- Added `e2e` pytest marker to split fast tests from Ollama-dependent ones

### Bug found
- `/health/full` called `await`-less on an async `OllamaClient.health_check()` → crashed with `TypeError: 'coroutine' object is not subscriptable`
- Only caught once API-level tests existed — pure unit tests never touched the route layer

### Learning
- Changing a constructor signature breaks nothing visibly until something instantiates it — only running tests catches it, not reading the diff
- Unit tests check logic; API tests check wiring (`Depends`, `app.state`, async/await) — need both
- Application-scoped services remove "which instance is this" bugs by making one-shared-instance structural, not a convention

### Limitations
- `/health/full` checks run sequentially, not in parallel
- No CI workflow yet runs the `e2e` split automatically

### Planned Improvements
- CI workflow: `pytest -m "not e2e"` per commit, `pytest -m e2e` daily
- Parallelize health checks
- Alembic migrations, evaluation layer (RAGAS)


---

## Chunk ID Collision Fix (v0.4.0)

### Bug found
- Chunk IDs were `chunk_{i}` per document, so later documents collided with earlier ones and Chroma silently skipped them — 97 of 313 chunks indexed, 3 of 5 PDFs had zero chunks; fixed by scoping IDs to `{document_id}_chunk_{chunk_index}`.
- Learning: a successful ingest call doesn't prove storage — reconcile vector store counts against SQL `chunk_count` before trusting any retrieval evaluation.

## Evaluation Harness

### first run (before fixing the question labels): hit@3 0.68 · hit@5 0.82 · hit@10 0.91 · MRR 0.58 · doc_hit@3 0.89
- Right chunk usually retrieved but ranked low (hit@10 ≫ hit@3) → ranking problem, not coverage.
- Right paper found more often than right passage (doc_hit@3 0.89 vs hit@3 0.68).
- Paraphrased questions land in the wrong paper more often (doc_hit@3 0.73 vs 0.97 direct).
- Own paper scores lowest (hit@3 0.40): overlapping papers and generic background chunks compete.
- No clean score threshold (unanswerable max 0.77 > relevant min 0.64) → "no answer" handled in generation.

###
- Table answers are extracted but retrieve poorly (gmm_q06, ras_q05 found in chunks 52/33, not in top 10):
  grids of names/numbers embed weakly. Candidate fix: table-aware chunking (caption + row-to-text).
- Chart labels are not extracted by OCR (oli_q07): known parsing limitation, kept in the set.

---

## Retrieval Experiments

Each experiment's hypothesis, setup, results and decision is recorded in
[`evaluation/retrieval/EXPERIMENTS.md`](evaluation/retrieval/EXPERIMENTS.md).

- E0, answer reachability: 43/44 answers exist in an indexed chunk → ranking, not ingestion, is the bottleneck.
- E1, BGE query prefix: not adopted, no effect (hit@3 30 → 31/44, one question).
- E2, neighbour dedup: not adopted, tie at k=3; answers moved up consistently (hit@5 35 → 37), but dropping "duplicates" at query time also drops unique text (bic_q05) → fix overlap in chunking instead (E4, after the reranker in E3).
- Learning: with 44 answerable questions, 1 question = 2.3 points, so ±1–2 questions is noise. Judge by the Improved/Regressed lists, not aggregate deltas.
