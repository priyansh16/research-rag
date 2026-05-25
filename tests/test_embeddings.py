import numpy as np


def test_embedding_generation(
    embedding_service
):

    embedding = embedding_service.embed_text(
        "Machine Learning"
    )

    assert isinstance(embedding, list)

    assert len(embedding) > 0


def test_embedding_dimensions_consistent(
    embedding_service
):

    emb1 = embedding_service.embed_text(
        "Machine Learning"
    )

    emb2 = embedding_service.embed_text(
        "Ice Hockey"
    )

    assert len(emb1) == len(emb2)


def test_embeddings_are_normalized(
    embedding_service
):

    embedding = embedding_service.embed_text(
        "Machine Learning"
    )

    norm = np.linalg.norm(embedding)

    assert abs(norm - 1.0) < 0.01