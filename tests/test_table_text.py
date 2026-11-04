"""
Unit tests for rebuilding table text from the PDF text layer.
Fakes only: no PDF, no unstructured model run.
"""

from types import SimpleNamespace

import pytest

from src.services.parsers import unstructured_parser
from src.services.parsers.table_text import (
    element_bbox_in_pdf_points,
    words_to_rows,
)
from src.services.parsers.unstructured_parser import elements_to_text


def w(x0, y0, text, width=None, height=10.0):
    """A PyMuPDF-style word tuple."""
    width = width if width is not None else 6.0 * len(text)
    return (x0, y0, x0 + width, y0 + height, text, 0, 0, 0)


# element_bbox_in_pdf_points
def test_bbox_scales_pixels_to_points():
    # 1700x2200 px render of a 612x792 pt page (≈ 200 dpi); no padding for exactness.
    bbox = element_bbox_in_pdf_points(
        [(170, 220), (170, 440), (850, 440), (850, 220)], 1700, 2200, 612, 792, padding=0
    )
    assert bbox == pytest.approx((61.2, 79.2, 306.0, 158.4))


def test_bbox_padding_is_clamped_to_page():
    bbox = element_bbox_in_pdf_points([(0, 0), (1700, 2200)], 1700, 2200, 612, 792, padding=5)
    assert bbox == (0.0, 0.0, 612, 792)


def test_bbox_rejects_empty_system():
    with pytest.raises(ValueError):
        element_bbox_in_pdf_points([(0, 0)], 0, 0, 612, 792)


# words_to_rows
def test_rows_grouped_by_line_and_ordered_left_to_right():
    words = [
        w(200, 100, "Växjö"), w(100, 100, "17/18"),           # same row, out of order
        w(100, 120, "18/19"), w(200, 121, "Frölunda"),        # slight vertical jitter
    ]
    assert words_to_rows(words) == ["17/18 | Växjö", "18/19 | Frölunda"]


def test_small_gap_joins_words_in_one_cell():
    # "Växjö (1, champions)": words 3 pt apart stay in one cell.
    words = [w(100, 100, "17/18", width=30), w(200, 100, "Växjö", width=30),
             w(233, 100, "(1,", width=15), w(251, 100, "champions)", width=50)]
    assert words_to_rows(words) == ["17/18 | Växjö (1, champions)"]


def test_rows_skip_blank_words_and_handle_empty():
    assert words_to_rows([]) == []
    assert words_to_rows([w(100, 100, "  ")]) == []


def test_explicit_column_gap():
    words = [w(100, 100, "a", width=10), w(115, 100, "b", width=10)]   # 5 pt gap
    assert words_to_rows(words, column_gap=4) == ["a | b"]
    assert words_to_rows(words, column_gap=6) == ["a b"]


# elements_to_text
class Table(SimpleNamespace):
    def __str__(self):
        return self.text


class NarrativeText(Table):
    pass


def element(cls, text, page=1, points=((0, 0), (10, 10))):
    coordinates = SimpleNamespace(points=points, system=SimpleNamespace(width=100, height=100))
    return cls(text=text, metadata=SimpleNamespace(page_number=page, coordinates=coordinates))


class FakePage:
    rect = SimpleNamespace(width=100, height=100)


class FakePdf:
    def __getitem__(self, index):
        return FakePage()

    def close(self):
        pass


@pytest.fixture
def fake_pdf(monkeypatch):
    import fitz
    monkeypatch.setattr(fitz, "open", lambda path: FakePdf())


def test_ocr_mode_keeps_old_output():
    elements = [element(NarrativeText, "Some text."), element(Table, "17/18 Vaxjo (1)")]
    assert elements_to_text(elements, table_source="ocr") == (
        "[NarrativeText] Some text.\n\n[Table] 17/18 Vaxjo (1)"
    )


def test_text_layer_mode_replaces_table_text(fake_pdf, monkeypatch):
    monkeypatch.setattr(unstructured_parser, "table_text_from_text_layer",
                        lambda page, bbox: "17/18 | Växjö (1)\n18/19 | Frölunda (3)")
    elements = [element(NarrativeText, "Some text."), element(Table, "17/18 Vaxjo (1)")]
    text = elements_to_text(elements, file_path="x.pdf", table_source="text_layer")
    assert text == "[NarrativeText] Some text.\n\n[Table] 17/18 | Växjö (1)\n18/19 | Frölunda (3)"


def test_text_layer_mode_falls_back_to_ocr_without_text_layer(fake_pdf, monkeypatch):
    monkeypatch.setattr(unstructured_parser, "table_text_from_text_layer", lambda page, bbox: None)
    text = elements_to_text([element(Table, "17/18 Vaxjo (1)")], file_path="x.pdf",
                            table_source="text_layer")
    assert text == "[Table] 17/18 Vaxjo (1)"


def test_text_layer_mode_falls_back_without_coordinates(fake_pdf):
    table = Table(text="17/18 Vaxjo (1)", metadata=SimpleNamespace(page_number=1, coordinates=None))
    assert elements_to_text([table], file_path="x.pdf", table_source="text_layer") == "[Table] 17/18 Vaxjo (1)"


def test_text_layer_mode_without_file_path_keeps_ocr():
    # conftest calls elements_to_text(elements) with no path: must not fail.
    assert elements_to_text([element(Table, "t")], table_source="text_layer") == "[Table] t"


def test_unknown_table_source_rejected():
    with pytest.raises(ValueError):
        elements_to_text([], table_source="html")
