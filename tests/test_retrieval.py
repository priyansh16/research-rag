from src.services.retrieval.retrieval_service import RetrievalService

def test_semantic_retrieval_ml():

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="machine learning research",
    )
    

    top_chunk = results[0]["chunks"]

    assert (
        "Gaussian Mixture Models"
        in top_chunk
        or
        "Machine Learning"
        in top_chunk
    )


def test_semantic_retrieval_hockey():

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="ice hockey analytics"
        )

    combined = " ".join(
        [
            r["chunks"]
            for r in results
        ]
    )

    assert (
        "Ice Hockey"
        in combined
    )


def test_retrieval_returns_scores():

    retrieval_service = RetrievalService()

    results = retrieval_service.search(
        query="Python"
    )

    assert "score" in results[0]

    assert isinstance(
        results[0]["score"],
        float
    )
    