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
