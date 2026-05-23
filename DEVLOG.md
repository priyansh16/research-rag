# Developer Log
1. Initial setup
    - clean folder structure, 
    - maintaed env with uv
    - created basic Fastapp APP
    - tested health points are reachable

2. PDF Extraction pipeline
    - implementattion
        - PDF ingestion pipeline
        - Extracted text using PyMuPDF (fitz)
        - Extracted tables using pdfplumber
        - Converted tables into RAG-friendly natural language
        - Built modular pipeline:
            - `extract_content()` → abstraction layer
            - `process_pdf()` → orchestration
        - Integrated with FastAPI upload endpoint
        - Stored extracted content into database
    - Learning:
        - PDF parsing is not trivial — layout matters more than text
        - Tables are hard to extract reliably with heuristic tools
        - Converting tables → natural language improves RAG compatibility
        - Separation of concerns (pipeline vs utils) is critical
    - Limitations
        - Table extraction is inconsistent for complex PDFs
        - Multi-column layouts may break text flow
        - No chunking yet → large text blobs
