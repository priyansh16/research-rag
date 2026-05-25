import fitz
import pdfplumber
import re
from typing import List
from loguru import logger

# === Extraction ===
def extract_text(file_path:str)->str:
    """
    Extract raw text from PDF using PyMuPDF.

    Why:
    - Fast
    - Works well for most academic PDFs

    Returns:
        str: raw extracted text
    """
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    return text

def extract_tables(file_path:str)->List[List[List[str]]]:
    """
    Extracts tables using pdfplumber.
    
    Returns:
        list of tables
        each table = list of rows
        each row = list of cells
    """
    tables =[]
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            
            for table in page_tables:
                if table:
                    tables.append(table)
        
    return tables
                       

# === Cleaning ===
def fix_hyphenation(text:str) -> str:
    """
    Fix words broken across lines.

    Example:
    "sum-\\nmarized" → "summarized"
    """
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    return text


def normalize_newlines(text: str) -> str:
    """
    Clean newline issues.

    - merge broken lines
    - preserve paragraphs
    """
    # Merge single line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize paragraph spacing
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text

def normalize_spaces(text: str) -> str:
    """
    Remove excessive spaces and tabs.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def remove_noise(text: str) -> str:
    """
    Remove noisy artifacts from PDF extraction.

    Examples:
    - repeated commas
    - broken tokens
    """
    text = re.sub(r"\s*,\s*", " ", text)
    return text

def remove_table_noise(text: str) -> str:
    """
    Remove raw table-like numeric lines from text.

    These will be replaced with structured table text.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # heuristic: lots of numbers → table
        if len(re.findall(r"\d", line)) > 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline.

    Order matters!
    """
    # text = remove_table_noise(text)
    text = fix_hyphenation(text)
    text = normalize_newlines(text)
    text = remove_noise(text)
    text = normalize_spaces(text)

    return text.strip()

# === Table Handling ===
def format_table(table: List[List[str]]) -> str:
    """
    Convert table into natural language.

    Example:
    Input:
        ["Situation", "Average GPIV", "Goals"]
        ["Reducing deficit", "0.249", "887"]

    Output:
        "Reducing deficit has Average GPIV 0.249 and Goals 887."
    """

    if not table or len(table) < 2:
        return ""

    headers = table[0]
    rows = table[1:]

    sentences = []

    for row in rows:
        if not row:
            continue

        pairs = []

        for h, v in zip(headers, row):
            if h and v:
                h_clean = h.strip()
                v_clean = v.strip()

                if h_clean and v_clean:
                    pairs.append(f"{h_clean} is {v_clean}")

        if pairs:
            sentences.append(", ".join(pairs) + ".")

    return " ".join(sentences)


def format_tables(tables: List[List[List[str]]]) -> str:
    """
    Process all tables into text block.

    Returns:
        str: combined table text
    """
    formatted = []
    
    for table in tables:
        table_text = format_table(table)

        if table_text:
            formatted.append("Table Data: " + table_text)

    return "\n\n".join(formatted)