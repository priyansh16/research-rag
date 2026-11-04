import re
from typing import List, Optional

from src.core.config import settings


# Element type tag the parser puts in front of every element, e.g. "[NarrativeText] ".
_ELEMENT_TAG = re.compile(r"^\[[A-Za-z]+\] ")
# Sentence end followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def estimate_size(text: str) -> int:
    """
    Rough character-based chunk size estimation.

    Later will be replaced with token counting.
    """
    return len(text)


def split_elements(text: str) -> List[str]:
    """
    Split structured parsed text into semantic elements.

    Elements are separated by double newlines.
    """

    return [
        e.strip()
        for e in text.split("\n\n")
        if e.strip()
        ]


def is_title(element: str) -> bool:
    """
    Detect whether an element is a title.
    """

    return element.startswith("[Title]")


def split_long_element(element: str, max_size: int) -> List[str]:
    """
    Split an element longer than max_size into pieces of at most max_size
    characters.

    Tables with line breaks (rebuilt from the text layer, one row per line)
    split between rows; everything else splits at sentence boundaries. A
    unit that is still too long (OCR'd tables have no sentences) is cut at the
    last space before the limit, or hard-cut if there is no space. The
    element's type tag ("[Table] ") is repeated on every piece so each chunk
    still says what kind of text it holds.
    """
    if len(element) <= max_size:
        return [element]

    tag_match = _ELEMENT_TAG.match(element)
    tag = tag_match.group(0) if tag_match else ""
    body = element[len(tag):]
    budget = max_size - len(tag)
    if budget < 1:
        raise ValueError("max_size is too small to hold the element tag")

    if tag == "[Table] " and "\n" in body:
        units, joiner = [line.strip() for line in body.split("\n")], "\n"
    else:
        units, joiner = _SENTENCE_END.split(body), " "

    pieces: List[str] = []
    current = ""

    for unit in units:
        while len(unit) > budget:
            if current:
                pieces.append(current)
                current = ""
            cut = unit.rfind(" ", 0, budget + 1)
            if cut <= 0:
                cut = budget
            pieces.append(unit[:cut].rstrip())
            unit = unit[cut:].lstrip()

        if not unit:
            continue
        if current and len(current) + len(joiner) + len(unit) > budget:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}{joiner}{unit}" if current else unit

    if current:
        pieces.append(current)

    return [tag + piece for piece in pieces]


def trim_overlap(overlap: List[str], max_chars: int) -> List[str]:
    """
    Keep the longest run of trailing elements whose total length is at most
    max_chars. An element longer than max_chars ends the run.
    """
    kept: List[str] = []
    total = 0
    for element in reversed(overlap):
        if total + len(element) > max_chars:
            break
        kept.insert(0, element)
        total += len(element)
    return kept


def build_semantic_chunks(
    elements: List[str],
    max_size: Optional[int] = None,
    overlap_elements: Optional[int] = None,
    overlap_max_chars: Optional[int] = None,
    split_long: Optional[bool] = None,
) -> List[str]:
    """
    Build semantic chunks while preserving document hierarchy.

    Strategy:
    - preserve titles as semantic anchors
    - group related content together
    - split only when chunk becomes too large
    - maintain overlap for retrieval continuity

    Parameters default to settings (MAX_CHUNK_SIZE, OVERLAP_ELEMENTS,
    CHUNK_OVERLAP_MAX_CHARS, CHUNK_SPLIT_LONG_ELEMENTS). With the default
    settings the output matches the original chunker, except that it no
    longer emits an empty chunk when the first element is oversized and
    overlap_elements=0 really means no overlap.
    """
    max_size = settings.MAX_CHUNK_SIZE if max_size is None else max_size
    overlap_elements = settings.OVERLAP_ELEMENTS if overlap_elements is None else overlap_elements
    overlap_max_chars = settings.CHUNK_OVERLAP_MAX_CHARS if overlap_max_chars is None else overlap_max_chars
    split_long = settings.CHUNK_SPLIT_LONG_ELEMENTS if split_long is None else split_long

    chunks = []

    current_chunk: List[str] = []
    current_length = 0

    for element in elements:

        element = element.strip()
        if not element:
            continue

        pieces = split_long_element(element, max_size) if split_long else [element]

        for piece in pieces:
            piece_length = len(piece)

            # Close the current chunk. Never close an empty one: an oversized
            # first element used to produce an empty chunk here.
            if current_chunk and current_length + piece_length > max_size:

                chunks.append("\n\n".join(current_chunk))

                # semantic overlap ([-0:] would be the whole list, so guard 0)
                overlap = current_chunk[-overlap_elements:] if overlap_elements > 0 else []

                if overlap_max_chars > 0:
                    overlap = trim_overlap(overlap, overlap_max_chars)
                    # Don't let carried-over text push the next chunk past the limit.
                    if sum(len(x) for x in overlap) + piece_length > max_size:
                        overlap = []

                current_chunk = overlap.copy()
                current_length = sum(len(x) for x in current_chunk)

            current_chunk.append(piece)
            current_length += piece_length

    # Final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks



def create_chunks(text: str) -> List[str]:
    """
    Main chunking pipeline.
    """

    elements = split_elements(text)

    chunks = build_semantic_chunks(elements)

    return chunks
