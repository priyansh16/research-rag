"""
Rebuild table text from the PDF's own text layer.

unstructured (hi_res) reads table cells by OCR'ing an image of the table, which
drops diacritics and garbles characters ("Växjö" -> "Vaxjo", "1st" -> "Ist"),
and str(table_element) flattens every cell into one line. The PDFs have a real
text layer, so for each table region we take the words PyMuPDF reports inside
the table's bounding box and lay them out again: one line per row, " | " where
a wide horizontal gap separates columns.

Pure helpers (bbox scaling, word -> row grouping) are separate from the PyMuPDF
call so they can be tested without a PDF.
"""

from statistics import median
from typing import List, Optional, Sequence, Tuple

# A PyMuPDF "words" entry: (x0, y0, x1, y1, text, block_no, line_no, word_no)
Word = Tuple[float, float, float, float, str, int, int, int]
Rect = Tuple[float, float, float, float]

# Fewer words than this inside a table box means no usable text layer there
# (e.g. a scanned page), so the caller falls back to the OCR text.
MIN_TABLE_WORDS = 4
# Padding (PDF points) around the table box, since layout boxes are approximate.
BBOX_PADDING = 2.0


def element_bbox_in_pdf_points(
    points: Sequence[Tuple[float, float]],
    system_width: float,
    system_height: float,
    page_width: float,
    page_height: float,
    padding: float = BBOX_PADDING,
) -> Rect:
    """
    Convert an unstructured element's corner points, given in its coordinate
    system (hi_res: pixels of the rendered page image), to a PDF-point
    rectangle (x0, y0, x1, y1) on the page, padded slightly.
    Both systems have their origin at the top-left corner.
    """
    if system_width <= 0 or system_height <= 0:
        raise ValueError("coordinate system has no size")
    scale_x = page_width / system_width
    scale_y = page_height / system_height
    xs = [x * scale_x for x, _ in points]
    ys = [y * scale_y for _, y in points]
    return (
        max(0.0, min(xs) - padding),
        max(0.0, min(ys) - padding),
        min(page_width, max(xs) + padding),
        min(page_height, max(ys) + padding),
    )


def words_to_rows(words: Sequence[Word], column_gap: Optional[float] = None) -> List[str]:
    """
    Group words into table rows by vertical position, left to right within a
    row, with " | " between words separated by more than column_gap points.

    column_gap defaults to 1.5x the median word height: wider than the space
    between words in a cell, narrower than a typical column gutter.
    """
    words = [w for w in words if w[4].strip()]
    if not words:
        return []

    heights = [w[3] - w[1] for w in words]
    row_tolerance = median(heights) / 2
    if column_gap is None:
        column_gap = 1.5 * median(heights)

    # Group by vertical centre: a word joins the current row if its centre is
    # within half a word height of the row's first word.
    by_centre = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: List[List[Word]] = []
    row_centre = None
    for word in by_centre:
        centre = (word[1] + word[3]) / 2
        if row_centre is None or abs(centre - row_centre) > row_tolerance:
            rows.append([])
            row_centre = centre
        rows[-1].append(word)

    lines = []
    for row in rows:
        row.sort(key=lambda w: w[0])
        parts = [row[0][4]]
        for previous, word in zip(row, row[1:]):
            gap = word[0] - previous[2]
            parts.append(" | " if gap > column_gap else " ")
            parts.append(word[4])
        lines.append("".join(parts))
    return lines


def table_text_from_text_layer(page, bbox: Rect) -> Optional[str]:
    """
    Table text rebuilt from the words inside bbox on a PyMuPDF page, one row
    per line. None when the region has no usable text layer.
    """
    import fitz

    words = page.get_text("words", clip=fitz.Rect(*bbox))
    if len(words) < MIN_TABLE_WORDS:
        return None
    rows = words_to_rows(words)
    return "\n".join(rows) if rows else None
