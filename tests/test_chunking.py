from src.core.config import settings
import re


def test_chunks_are_created(chunks):

    assert len(chunks) > 0


def test_chunks_respect_size_limit(chunks):

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