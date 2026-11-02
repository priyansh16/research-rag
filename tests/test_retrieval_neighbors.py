"""
Unit tests for drop_neighbor_chunks: pure logic, no model or ChromaDB.
"""

from src.services.retrieval.retrieval_service import drop_neighbor_chunks


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


def test_preserves_rank_order_and_stops_at_top_k():
    results = [r(1, 0), r(2, 0), r(3, 0), r(4, 0)]
    kept = drop_neighbor_chunks(results, window=2, top_k=2)
    assert ids(kept) == [(1, 0), (2, 0)]
    assert kept[0] is results[0]


def test_empty_results():
    assert drop_neighbor_chunks([], window=2, top_k=3) == []
