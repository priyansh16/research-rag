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

from src.services.vectorstore.chroma_service import (
    ChromaService
)

from src.services.retrieval.retrieval_service import (
    RetrievalService
)

from src.services.generation.ollama_client import (
    OllamaClient
)

from src.services.generation.generation_service import (
    GenerationService
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


@pytest.fixture(scope="session")
def chroma_service():
    """
    Shared ChromaDB client, pointed at the project's persistent store.
    """

    return ChromaService()


@pytest.fixture(scope="session")
def retrieval_service(
    embedding_service,
    chroma_service
):
    """
    Shared retrieval service, wired with the real embedding and
    vector-store dependencies.
    """

    return RetrievalService(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
    )


@pytest.fixture(scope="session")
def ollama_client():
    """
    Shared Ollama client. Only reachable when `ollama serve` is running
    locally — tests that need a live call skip automatically otherwise.
    """

    return OllamaClient()


@pytest.fixture(scope="session")
def generation_service(
    retrieval_service,
    guardrails,
    ollama_client
):
    """
    Shared generation service, wired with its real dependencies.
    """

    return GenerationService(
        retrieval_service=retrieval_service,
        guardrails_service=guardrails,
        ollama_client=ollama_client,
    )