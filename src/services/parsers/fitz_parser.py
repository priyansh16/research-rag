from src.services.parsers.pdf_utils import extract_text, extract_tables, format_tables, clean_text


def extract_with_fitz(file_path: str) -> str:
    """
    Inital extraction pipeline using pdfplumber and fitz.

    This orchestrates:
    - text extraction
    - table extraction
    - cleaning
    - merging
    
    can be called by pipeline.
    """

    raw_text = extract_text(file_path)
    
    tables = extract_tables(file_path)
    table_text = format_tables(tables)

    cleaned_text = clean_text(raw_text)

    final_text = cleaned_text + "\n\n" + table_text

    return final_text

