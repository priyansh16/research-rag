"""
Unit tests for the E4 chunking options: overlap cap, splitting long elements,
and the empty-chunk fix. Pure logic: no PDF, no model, no ChromaDB.
"""

import re

import pytest

from src.services.chunking import (
    build_semantic_chunks,
    split_long_element,
    trim_overlap,
)


def old_build_semantic_chunks(elements, max_size, overlap_elements):
    """The chunker before E4, copied verbatim (minus settings) as a regression reference."""
    chunks, current_chunk, current_length = [], [], 0
    for element in elements:
        element = element.strip()
        if not element:
            continue
        if current_length + len(element) > max_size:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = current_chunk[-overlap_elements:].copy()
            current_length = sum(len(x) for x in current_chunk)
        current_chunk.append(element)
        current_length += len(element)
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


def para(n, i):
    """An element of exactly n characters with a recognisable id."""
    head = f"[NarrativeText] p{i} "
    return head + "x" * (n - len(head))


# A realistic mix: short titles, mid paragraphs, one long table, one long paragraph.
ELEMENTS = [
    "[Title] 1 Introduction",
    para(400, 1), para(700, 2), para(300, 3),
    "[Title] 2 Method",
    para(900, 4), para(200, 5),
    "[Table] " + " ".join(f"row{i} 1.234 5.678" for i in range(150)),
    para(500, 6),
    "[NarrativeText] " + " ".join(f"Sentence number {i} ends here." for i in range(80)),
    para(250, 7), "[Title] 3 Results", para(650, 8),
]


# Defaults keep the old behaviour
def test_old_mode_matches_previous_chunker():
    # First element is short, so the old empty-chunk bug doesn't trigger.
    new = build_semantic_chunks(ELEMENTS, max_size=1200, overlap_elements=2,
                                overlap_max_chars=0, split_long=False)
    assert new == old_build_semantic_chunks(ELEMENTS, max_size=1200, overlap_elements=2)


def test_no_empty_chunk_when_first_element_is_oversized():
    elements = [para(1500, 1), para(100, 2)]
    chunks = build_semantic_chunks(elements, max_size=1200, overlap_elements=2,
                                   overlap_max_chars=0, split_long=False)
    assert "" not in chunks
    assert old_build_semantic_chunks(elements, 1200, 2)[0] == ""   # the bug being fixed


def test_zero_overlap_elements_means_no_overlap():
    elements = [para(500, i) for i in range(6)]
    chunks = build_semantic_chunks(elements, max_size=1200, overlap_elements=0,
                                   overlap_max_chars=0, split_long=False)
    pieces = [p for c in chunks for p in c.split("\n\n")]
    assert len(pieces) == len(elements)


# split_long_element
def test_split_short_element_unchanged():
    assert split_long_element("[Title] Intro", 1200) == ["[Title] Intro"]


def test_split_at_sentence_boundaries_and_keep_tag():
    element = "[NarrativeText] " + " ".join(f"Sentence {i} ends here." for i in range(100))
    pieces = split_long_element(element, 300)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300
        assert piece.startswith("[NarrativeText] ")
        assert piece.endswith(".")          # never cuts mid-sentence when it doesn't have to


def test_split_preserves_all_text():
    element = "[NarrativeText] " + " ".join(f"Sentence {i} ends here." for i in range(100))
    pieces = split_long_element(element, 300)
    rejoined = " ".join(p[len("[NarrativeText] "):] for p in pieces)
    assert rejoined.split() == element[len("[NarrativeText] "):].split()


def test_split_table_without_sentences_on_spaces():
    element = "[Table] " + " ".join(f"row{i} 1.234 5.678" for i in range(150))
    pieces = split_long_element(element, 400)
    assert all(len(p) <= 400 and p.startswith("[Table] ") for p in pieces)
    tokens = [t for p in pieces for t in p[len("[Table] "):].split()]
    assert tokens == element[len("[Table] "):].split()     # no token cut in half


def test_split_text_layer_table_between_rows():
    rows = [f"{i:02d}/{i + 1:02d} | Växjö ({i}, champions) | HV71 (quarter-final)" for i in range(40)]
    element = "[Table] " + "\n".join(rows)
    pieces = split_long_element(element, 400)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 400 and piece.startswith("[Table] ")
        for line in piece[len("[Table] "):].split("\n"):
            assert line in rows                      # whole rows only, never cut mid-row
    assert [l for p in pieces for l in p[len("[Table] "):].split("\n")] == rows


def test_narrative_with_line_break_still_splits_on_sentences():
    element = "[NarrativeText] " + "\n".join(f"Sentence {i} ends here." for i in range(60))
    pieces = split_long_element(element, 300)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300
        assert "\n" not in piece          # sentence mode re-joins with spaces
        assert piece.endswith(".")


def test_split_hard_cuts_a_single_huge_word():
    element = "[Text] " + "a" * 1000
    pieces = split_long_element(element, 300)
    assert all(len(p) <= 300 for p in pieces)
    assert "".join(p[len("[Text] "):] for p in pieces) == "a" * 1000


def test_split_untagged_element():
    pieces = split_long_element("word " * 200, 120)
    assert all(len(p) <= 120 for p in pieces)


def test_split_rejects_max_size_smaller_than_tag():
    with pytest.raises(ValueError):
        split_long_element("[NarrativeText] " + "x" * 50, 10)


# trim_overlap
def test_trim_overlap_keeps_trailing_elements_under_cap():
    assert trim_overlap(["a" * 100, "b" * 100, "c" * 100], 250) == ["b" * 100, "c" * 100]


def test_trim_overlap_stops_at_long_element():
    # The long middle element ends the run even though "a" would fit.
    assert trim_overlap(["a" * 10, "b" * 500, "c" * 100], 300) == ["c" * 100]


def test_trim_overlap_empty_when_last_element_too_long():
    assert trim_overlap(["a" * 10, "b" * 500], 300) == []


# E4 mode end to end
E4 = dict(max_size=1200, overlap_elements=1, overlap_max_chars=300, split_long=True)


def joined_limit(chunk, max_size):
    """Element lengths count toward max_size; the "\\n\\n" joins between them don't."""
    return max_size + 2 * chunk.count("\n\n")


def test_e4_chunks_respect_size_limit():
    for chunk in build_semantic_chunks(ELEMENTS, **E4):
        assert len(chunk) <= joined_limit(chunk, 1200)


def test_e4_old_mode_exceeds_limit_on_same_input():
    # Guards the test above: the same input really does produce oversized chunks without E4.
    old = build_semantic_chunks(ELEMENTS, max_size=1200, overlap_elements=2,
                                overlap_max_chars=0, split_long=False)
    assert any(len(c) > joined_limit(c, 1200) for c in old)


def test_e4_overlap_only_carries_short_elements():
    chunks = build_semantic_chunks(ELEMENTS, **E4)
    for previous, current in zip(chunks, chunks[1:]):
        first = current.split("\n\n")[0]
        if first in previous.split("\n\n"):       # carried over
            assert len(first) <= 300


def test_e4_every_element_is_kept():
    chunks = build_semantic_chunks(ELEMENTS, **E4)
    text = " ".join(chunks)
    for i in range(1, 9):
        assert re.search(rf"\bp{i}\b", text)
    assert "row149" in text and "Sentence number 79" in text


def test_e4_no_chunk_is_only_overlap():
    chunks = build_semantic_chunks(ELEMENTS, **E4)
    for previous, current in zip(chunks, chunks[1:]):
        assert not set(current.split("\n\n")) <= set(previous.split("\n\n"))


def test_e4_empty_input():
    assert build_semantic_chunks([], **E4) == []
