from typing import List
from src.core.config import settings



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


def build_semantic_chunks(elements: List[str]) -> List[str]:
    """
    Build semantic chunks while preserving document hierarchy.

    Strategy:
    - preserve titles as semantic anchors
    - group related content together
    - split only when chunk becomes too large
    - maintain overlap for retrieval continuity
    """

    chunks = []

    current_chunk = []
    current_length = 0

    for element in elements:
        
        element = element.strip()
        # Track semantic titles
        if not element:
            continue
        
        element_length = len(element)
        
        # create chunk 
        if current_length + element_length > settings.MAX_CHUNK_SIZE:
            
            chunks.append("\n\n".join(current_chunk))
            
            # semantic overlap
            overlap = current_chunk[-settings.OVERLAP_ELEMENTS:]
            
            current_chunk = overlap.copy()
            current_length = sum(len(x) for x in current_chunk)
        
        current_chunk.append(element)
        current_length += element_length     
      
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