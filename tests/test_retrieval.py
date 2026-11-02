import pytest


def test_semantic_retrieval_ml(retrieval_service):

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


def test_semantic_retrieval_hockey(retrieval_service):

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


def test_retrieval_returns_scores(retrieval_service):

    results = retrieval_service.search(
        query="Python"
    )

    assert "score" in results[0]

    assert isinstance(
        results[0]["score"],
        float
    )
    

def test_retrieval_respects_top_k(retrieval_service):

    results = retrieval_service.search(
        query="ice hockey analytics",
        top_k=5
    )

    assert len(results) == 5


def test_retrieval_defaults_to_settings_top_k(retrieval_service):

    from src.core.config import settings

    results = retrieval_service.search(
        query="ice hockey analytics"
    )

    assert len(results) == settings.TOP_K_RESULTS


def test_retrieval_rejects_invalid_top_k(retrieval_service):

    with pytest.raises(ValueError):
        retrieval_service.search(
            query="ice hockey analytics",
            top_k=0
        )
