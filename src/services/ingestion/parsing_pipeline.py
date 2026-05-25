from src.core.config import settings

from src.services.parsers.fitz_parser import extract_with_fitz
from src.services.parsers.unstructured_parser import extract_with_unstructured


def extract_content(file_path: str) -> str:
    """
    Main extraction abstraction layer.
    """

    if settings.PARSER_BACKEND == "fitz":
        return extract_with_fitz(file_path)

    elif settings.PARSER_BACKEND == "unstructured":
        return extract_with_unstructured(file_path)

    raise ValueError(f"Unsupported parser backend: {settings.PARSER_BACKEND}")
