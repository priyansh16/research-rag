from unstructured.partition.pdf import partition_pdf
from loguru import logger


def extract_elements(file_path: str):
    """
    Extract semantic document elements from PDF.

    Unstructured identifies:
    - Titles
    - Narrative text
    - Tables
    - Lists
    - Headers
    - etc.

    Returns:
        list: semantic document elements
    """

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
    )

    logger.info(f"Extracted {len(elements)} semantic elements")

    return elements

def elements_to_text(elements):
    """
    Convert semantic elements into structured text.

    Preserves:
    - titles
    - paragraphs
    - tables
    """

    structured_text = []

    for element in elements:
        element_type = type(element).__name__

        text = str(element).strip()

        if not text:
            continue

        structured_text.append(
            f"[{element_type}] {text}"
        )

    return "\n\n".join(structured_text)

def extract_with_unstructured(file_path:str) -> str:
    """
    parse document with unstructured
    called by pipeline
    """
    elements = extract_elements(file_path)
    
    structured_text = elements_to_text(elements)
    return structured_text