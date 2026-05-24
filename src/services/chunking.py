import re
from typing import List


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs.

    Paragraph = separated by double newline.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_into_sentences(text: str) -> List[str]:
    """
    Basic sentence splitter.

    Not perfect but good for now.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Recursive chunking with overlap.

    Strategy:
    - paragraph → sentence → chunk
    """

    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())

            # overlap
            current_chunk = current_chunk[-overlap:] + " " + para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def separate_tables(text: str):
    """
    Split text into:
    - normal content
    - table content
    """

    sections = text.split("Table Data:")

    main_text = sections[0]
    tables = sections[1:] if len(sections) > 1 else []

    return main_text, tables

def create_chunks(text: str):
    """
    Full chunking pipeline.
    """

    main_text, tables = separate_tables(text)

    text_chunks = chunk_text(main_text)

    table_chunks = [f"Table Data: {t.strip()}" for t in tables]

    return text_chunks + table_chunks

