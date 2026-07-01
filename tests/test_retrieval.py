from src.services.retrieval.retrieval_service import RetrievalService

def test_semantic_retrieval_ml( embedded_chunks):

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="machine learning research",
        chunk_records=embedded_chunks
    )

    top_chunk = results[0]["chunk"]

    assert (
        "Gaussian Mixture Models"
        in top_chunk
        or
        "Machine Learning"
        in top_chunk
    )


def test_semantic_retrieval_cisco(
    embedded_chunks
):

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="software engineering experience",
        chunk_records=embedded_chunks
    )

    combined = " ".join(
        [
            r["chunk"]
            for r in results
        ]
    )

    assert "Cisco Systems" in combined


def test_semantic_retrieval_hockey(
    embedded_chunks
):

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="ice hockey analytics",
        chunk_records=embedded_chunks
    )

    combined = " ".join(
        [
            r["chunk"]
            for r in results
        ]
    )

    assert (
        "Ice Hockey"
        in combined
    )


def test_retrieval_returns_scores(
    embedded_chunks
):

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="Python",
        chunk_records=embedded_chunks
    )

    assert "score" in results[0]

    assert isinstance(
        results[0]["score"],
        float
    )
    