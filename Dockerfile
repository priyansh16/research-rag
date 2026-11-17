# RAG API image: FastAPI + embedding model + reranker + unstructured PDF parsing.
# Ollama runs in its own container (see compose.yaml) or on the host
# (compose.host-ollama.yaml).

FROM python:3.11-slim

# System libraries for unstructured[pdf] hi_res parsing:
#   poppler-utils  PDF rendering (pdf2image)
#   tesseract-ocr  OCR for tables and scanned text
#   libgl1, libglib2.0-0  OpenCV, used by the layout model
#   libmagic1      file type detection
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils tesseract-ocr libgl1 libglib2.0-0 libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Same uv version as local development.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# spaCy model used by unstructured's PDF text extraction. unstructured installs it
# into site-packages on first use, which fails for the non-root user (site-packages
# is owned by root); it then silently skips the PDF text layer and OCRs every page.
# Pre-installed here with the version and hash unstructured pins in
# unstructured/nlp/tokenize.py; update both if unstructured changes its pin.
RUN --mount=type=cache,target=/root/.cache/uv \
    echo "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl --hash=sha256:1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85" \
        > /tmp/spacy-model.txt \
    && uv pip install --python /opt/venv/bin/python --require-hashes -r /tmp/spacy-model.txt \
    && rm /tmp/spacy-model.txt \
    && python -c "import spacy; spacy.load('en_core_web_sm')"

COPY src ./src

# Non-root user. The directories exist in the image with this owner, so the
# named volumes mounted on them start out writable by the app.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/storage /app/data /app/logs /models/hf \
    && chown -R app:app /app/storage /app/data /app/logs /models
USER app

EXPOSE 8000

# Liveness only; /api/v1/health/full also checks Ollama, Chroma and the models.
# The first start downloads the embedding model and the reranker, hence the long start period.
# --start-interval is set too: otherwise Docker checks every 5 s during the start period.
HEALTHCHECK --interval=60s --start-interval=60s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=4)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
