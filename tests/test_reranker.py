"""
Unit tests for the cross-encoder reranker and its use in RetrievalService.search.
Fakes only: no model download, no ChromaDB.
"""

import pytest

from src.core.config import settings
from src.services.retrieval.reranker_service import (
    RerankerService,
    apply_rerank_scores,
    build_reranker_service,
)
from src.services.retrieval.retrieval_service import RetrievalService


def r(name, score):
    return {"chunks": name, "metadata": {"chunk_index": name}, "score": score}


class FakeCrossEncoder:
    """Scores a (query, chunk) pair from a lookup table keyed by chunk text."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores_by_text[text] for _, text in pairs]


class FakeEmbeddingService:
    def embed_text(self, text):
        return [0.0]


class FakeChromaService:
    """Returns `n_stored` chunks named c0, c1, ... with falling similarity."""

    def __init__(self, n_stored=30):
        self.n_stored = n_stored
        self.requested_top_k = None

    def search(self, query_embeddings, top_k):
        self.requested_top_k = top_k
        n = min(top_k, self.n_stored)
        return {
            "documents": [[f"c{i}" for i in range(n)]],
            "metadatas": [[{"document_id": 1, "chunk_index": i} for i in range(n)]],
            "distances": [[i / 100 for i in range(n)]],
        }


# apply_rerank_scores
def test_apply_rerank_scores_sorts_by_reranker_score():
    results = [r("a", 0.9), r("b", 0.8), r("c", 0.7)]
    reranked = apply_rerank_scores(results, [0.1, 0.9, 0.5])
    assert [x["chunks"] for x in reranked] == ["b", "c", "a"]


def test_apply_rerank_scores_keeps_vector_score():
    reranked = apply_rerank_scores([r("a", 0.9)], [0.3])
    assert reranked[0]["score"] == 0.3
    assert reranked[0]["vector_score"] == 0.9


def test_apply_rerank_scores_does_not_mutate_input():
    results = [r("a", 0.9)]
    apply_rerank_scores(results, [0.3])
    assert results[0] == r("a", 0.9)


def test_apply_rerank_scores_ties_keep_vector_order():
    results = [r("a", 0.9), r("b", 0.8), r("c", 0.7)]
    reranked = apply_rerank_scores(results, [0.5, 0.5, 0.5])
    assert [x["chunks"] for x in reranked] == ["a", "b", "c"]


def test_apply_rerank_scores_rejects_length_mismatch():
    with pytest.raises(ValueError):
        apply_rerank_scores([r("a", 0.9)], [0.1, 0.2])


# RerankerService
def test_rerank_passes_query_chunk_pairs():
    model = FakeCrossEncoder({"a": 0.1, "b": 0.9})
    reranker = RerankerService("fake", model=model)
    reranked = reranker.rerank("q", [r("a", 0.9), r("b", 0.8)])
    assert model.calls == [[("q", "a"), ("q", "b")]]
    assert [x["chunks"] for x in reranked] == ["b", "a"]


def test_rerank_empty_results_skips_model():
    model = FakeCrossEncoder({})
    assert RerankerService("fake", model=model).rerank("q", []) == []
    assert model.calls == []


def test_build_reranker_service_none_when_model_empty(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANKER_MODEL", "")
    assert build_reranker_service() is None


# RetrievalService.search with and without a reranker
def make_service(reranker=None, n_stored=30):
    chroma = FakeChromaService(n_stored=n_stored)
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        chroma_service=chroma,
        reranker_service=reranker,
    )
    return service, chroma


def test_search_without_reranker_fetches_k_and_keeps_vector_order():
    service, chroma = make_service()
    results = service.search("q", top_k=3)
    assert chroma.requested_top_k == 3
    assert [x["chunks"] for x in results] == ["c0", "c1", "c2"]
    assert "vector_score" not in results[0]


def test_search_with_reranker_fetches_pool_and_returns_top_k(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_CANDIDATES", 20)
    # Reranker prefers later chunks: c19 best, c0 worst.
    model = FakeCrossEncoder({f"c{i}": i / 100 for i in range(30)})
    service, chroma = make_service(RerankerService("fake", model=model))

    results = service.search("q", top_k=3)

    assert chroma.requested_top_k == 20
    assert len(model.calls[0]) == 20
    assert [x["chunks"] for x in results] == ["c19", "c18", "c17"]
    assert results[0]["score"] == pytest.approx(0.19)
    assert results[0]["vector_score"] == pytest.approx(1 - 0.19)


def test_search_with_reranker_pool_never_smaller_than_k(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_CANDIDATES", 5)
    model = FakeCrossEncoder({f"c{i}": 0.5 for i in range(30)})
    service, chroma = make_service(RerankerService("fake", model=model))

    results = service.search("q", top_k=10)

    assert chroma.requested_top_k == 10
    assert len(results) == 10


def test_search_with_reranker_drops_neighbors_in_reranked_order(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_CANDIDATES", 15)
    monkeypatch.setattr(settings, "RETRIEVAL_NEIGHBOR_WINDOW", 1)
    # Reranker prefers later chunks: reranked order is c14, c13, c12, ...
    model = FakeCrossEncoder({f"c{i}": i / 100 for i in range(30)})
    service, chroma = make_service(RerankerService("fake", model=model))

    results = service.search("q", top_k=3)

    # Pool stays the reranker's, not k * (2*window + 1).
    assert chroma.requested_top_k == 15
    # Neighbours of kept chunks are dropped after reranking: 13 and 11 go.
    assert [x["chunks"] for x in results] == ["c14", "c12", "c10"]


def test_search_with_reranker_fewer_stored_than_pool(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RERANK_CANDIDATES", 20)
    model = FakeCrossEncoder({f"c{i}": i for i in range(4)})
    service, _ = make_service(RerankerService("fake", model=model), n_stored=4)

    results = service.search("q", top_k=3)

    assert [x["chunks"] for x in results] == ["c3", "c2", "c1"]
