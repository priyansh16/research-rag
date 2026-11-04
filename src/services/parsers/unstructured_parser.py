from typing import Optional

from unstructured.partition.pdf import partition_pdf
from loguru import logger

from src.core.config import settings
from src.services.parsers.table_text import (
    element_bbox_in_pdf_points,
    table_text_from_text_layer,
)


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

def _table_text_from_pdf(element, pdf) -> Optional[str]:
    """
    Rebuild a Table element's text from the PDF text layer (see table_text.py).
    None when the element has no page/coordinates or the region has no text layer.
    """
    metadata = element.metadata
    coordinates = getattr(metadata, "coordinates", None)
    page_number = getattr(metadata, "page_number", None)
    if coordinates is None or not coordinates.points or not page_number:
        return None

    page = pdf[page_number - 1]
    try:
        bbox = element_bbox_in_pdf_points(
            coordinates.points,
            coordinates.system.width,
            coordinates.system.height,
            page.rect.width,
            page.rect.height,
        )
    except (AttributeError, ValueError):
        return None
    return table_text_from_text_layer(page, bbox)


def elements_to_text(elements, file_path: Optional[str] = None, table_source: Optional[str] = None):
    """
    Convert semantic elements into structured text.

    Preserves:
    - titles
    - paragraphs
    - tables

    table_source (default settings.TABLE_TEXT_SOURCE):
    - "ocr": table text as unstructured returns it (OCR'd cells, flattened)
    - "text_layer": tables rebuilt from the PDF text layer, one row per line;
      falls back to the OCR text per table when there is no text layer.
      Needs file_path.
    """
    table_source = settings.TABLE_TEXT_SOURCE if table_source is None else table_source
    if table_source not in ("ocr", "text_layer"):
        raise ValueError(f"Unsupported TABLE_TEXT_SOURCE: {table_source}")

    pdf = None
    if table_source == "text_layer" and file_path:
        import fitz
        pdf = fitz.open(file_path)

    structured_text = []
    tables_from_text_layer = tables_from_ocr = 0

    try:
        for element in elements:
            element_type = type(element).__name__

            text = str(element).strip()

            if element_type == "Table" and pdf is not None:
                rebuilt = _table_text_from_pdf(element, pdf)
                if rebuilt:
                    text = rebuilt
                    tables_from_text_layer += 1
                else:
                    tables_from_ocr += 1

            if not text:
                continue

            structured_text.append(
                f"[{element_type}] {text}"
            )
    finally:
        if pdf is not None:
            pdf.close()

    if pdf is not None:
        logger.info(
            "Table text | from_text_layer={} | fell_back_to_ocr={}",
            tables_from_text_layer,
            tables_from_ocr,
        )

    return "\n\n".join(structured_text)

def extract_with_unstructured(file_path:str) -> str:
    """
    parse document with unstructured
    called by pipeline
    """
    elements = extract_elements(file_path)

    structured_text = elements_to_text(elements, file_path=file_path)
    return structured_text