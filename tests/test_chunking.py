from src.core.config import settings
from src.services.chunking import build_semantic_chunks
import re


def test_chunks_respect_size_limit(chunks):

    assert chunks

    for chunk in chunks:

        assert len(chunk) <= (
            settings.MAX_CHUNK_SIZE + 300
        )


def test_chunks_preserve_semantic_content(chunks):

    combined = " ".join(chunks)

    assert "Gaussian Mixture Models" in combined

    assert "Machine Learning" in combined


def test_chunks_do_not_contain_corrupted_words(chunks):

    corrupted_patterns = [
        r"\bostgreSQL\b",
    ]

    for chunk in chunks:

        for pattern in corrupted_patterns:

            assert re.search(pattern, chunk) is None


def test_chunks_contain_overlap(chunks):

    repeated_content = False

    for i in range(len(chunks) - 1):

        current_words = set(chunks[i].split())
        next_words = set(chunks[i + 1].split())

        overlap = current_words.intersection(next_words)

        if len(overlap) > 20:
            repeated_content = True

    assert repeated_content

# Regression tests for two chunker bugs found in E4. Pure logic: no PDF.
def para(n, i):
    head = f"[NarrativeText] p{i} "
    return head + "x" * (n - len(head))


def test_no_empty_chunk_when_first_element_is_oversized(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CHUNK_SIZE", 1200)
    monkeypatch.setattr(settings, "OVERLAP_ELEMENTS", 2)
    chunks = build_semantic_chunks([para(1500, 1), para(100, 2)])
    assert "" not in chunks
    assert chunks[0] == para(1500, 1)        # oversized element becomes its own first chunk


def test_zero_overlap_elements_means_no_overlap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CHUNK_SIZE", 1200)
    monkeypatch.setattr(settings, "OVERLAP_ELEMENTS", 0)
    elements = [para(500, i) for i in range(6)]
    chunks = build_semantic_chunks(elements)
    pieces = [p for c in chunks for p in c.split("\n\n")]
    assert pieces == elements          # every element exactly once


def test_default_overlap_still_repeats_last_elements(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CHUNK_SIZE", 1200)
    monkeypatch.setattr(settings, "OVERLAP_ELEMENTS", 2)
    elements = [para(500, i) for i in range(4)]
    chunks = build_semantic_chunks(elements)
    # The fixes don't change normal overlap: the last 2 elements are carried into the next chunk.
    assert chunks == [
        "\n\n".join(elements[0:2]),
        "\n\n".join(elements[0:3]),
        "\n\n".join(elements[1:4]),
    ]
