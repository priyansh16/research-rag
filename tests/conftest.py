import pytest

from src.services.parsers.unstructured_parser import (
    extract_elements,
    elements_to_text
)

from src.services.chunking import (
    create_chunks
)

from src.services.embeddings.embedding_service import (
    EmbeddingService
)

from src.services.guardrails.guardrails_service import (
    GuardrailsService
)


TEST_PDF_PATH = "tests/data/Priyansh_CV.pdf"

@pytest.fixture(scope="session")
def parsed_text():
    """
    Parse PDF once for all tests.
    """

    elements = extract_elements(TEST_PDF_PATH)

    text = elements_to_text(elements)

    return text


@pytest.fixture(scope="session")
def chunks(parsed_text):
    """
    Generate chunks once for all tests.
    """

    return create_chunks(parsed_text)


@pytest.fixture(scope="session")
def embedding_service():
    """
    Shared embedding service.
    """

    return EmbeddingService()

@pytest.fixture(scope="session")
def embedded_chunks(
    chunks,
    embedding_service
):
    """
    Generate embeddings for chunks.
    """

    embeddings = embedding_service.embed_texts(
        chunks
    )

    records = []

    for chunk, embedding in zip(chunks, embeddings):

        records.append(
            {
                "chunk": chunk,
                "embedding": embedding
            }
        )

    return records

@pytest.fixture(scope="session")
def guardrails():
    return GuardrailsService()