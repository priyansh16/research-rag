"""
RetrievalService tests.

- search() contract: real embedding model + the project's ChromaDB (conftest).
  Retrieval *quality* is measured by evaluation/retrieval, not here.
- drop_neighbor_chunks: pure logic, no model or ChromaDB.
"""

import pytest

from src.core.config import settings
from src.services.retrieval.retrieval_service import drop_neighbor_chunks


# search() contract
def test_retrieval_smoke_finds_hockey_content(retrieval_service):
    results = retrieval_service.search(query="ice hockey analytics")
    assert "Ice Hockey" in " ".join(r["chunks"] for r in results)


def test_retrieval_returns_scores(retrieval_service):
    results = retrieval_service.search(query="Python")
    assert isinstance(results[0]["score"], float)


def test_retrieval_respects_top_k(retrieval_service):
    results = retrieval_service.search(query="ice hockey analytics", top_k=5)
    assert len(results) == 5


def test_retrieval_defaults_to_settings_top_k(retrieval_service):
    results = retrieval_service.search(query="ice hockey analytics")
    assert len(results) == settings.TOP_K_RESULTS


def test_retrieval_rejects_invalid_top_k(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.search(query="ice hockey analytics", top_k=0)


# drop_neighbor_chunks
def r(doc_id, index, score=0.5):
    return {"chunks": f"{doc_id}-{index}", "metadata": {"document_id": doc_id, "chunk_index": index},
            "score": score}


def ids(results):
    return [(x["metadata"]["document_id"], x["metadata"]["chunk_index"]) for x in results]


def test_window_zero_only_truncates():
    results = [r(1, 26), r(1, 27), r(1, 25), r(2, 3)]
    assert ids(drop_neighbor_chunks(results, window=0, top_k=3)) == [(1, 26), (1, 27), (1, 25)]


def test_drops_neighbors_of_higher_ranked_chunks():
    # Real pattern from the baseline: 26, 27, 25 filled the top 3.
    results = [r(1, 26), r(1, 27), r(1, 25), r(1, 30), r(2, 4), r(1, 29)]
    assert ids(drop_neighbor_chunks(results, window=2, top_k=3)) == [(1, 26), (1, 30), (2, 4)]


def test_window_boundary_is_inclusive():
    results = [r(1, 10), r(1, 12), r(1, 13)]
    assert ids(drop_neighbor_chunks(results, window=2, top_k=3)) == [(1, 10), (1, 13)]


def test_neighbors_are_per_document():
    results = [r(1, 5), r(2, 5), r(2, 6), r(3, 6)]
    assert ids(drop_neighbor_chunks(results, window=1, top_k=4)) == [(1, 5), (2, 5), (3, 6)]


def test_only_kept_chunks_suppress():
    # 11 is dropped as a neighbour of 10, so it does not suppress 12.
    results = [r(1, 10), r(1, 11), r(1, 12)]
    assert ids(drop_neighbor_chunks(results, window=1, top_k=3)) == [(1, 10), (1, 12)]
