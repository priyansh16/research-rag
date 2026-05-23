from src.services.pdf_utils import extract_text, extract_tables, format_tables, clean_text


def extract_content(file_path:str) -> str:
    """
    Abstract extraction layer.
    Switch beetwen extraction logic:
    - PyMuPDF (current)
    - Unstructured(future)
    """
    return process_pdf(file_path)

def process_pdf(file_path: str) -> str:
    """
    Inital extraction pipeline using pdfplumber and fitz.

    This orchestrates:
    - text extraction
    - table extraction
    - cleaning
    - merging

    Directly called by API.
    """

    raw_text = extract_text(file_path)
    
    tables = extract_tables(file_path)
    table_text = format_tables(tables)

    cleaned_text = clean_text(raw_text)

    final_text = cleaned_text + "\n\n" + table_text

    return final_text

