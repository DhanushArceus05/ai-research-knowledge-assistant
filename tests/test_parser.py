import pytest

from app.document_processing.parser import PDFParser
from app.core.exceptions import ProcessingError


def test_parser_raises_on_invalid_pdf_path(tmp_path):
    parser = PDFParser()
    fake_pdf = tmp_path / "not_a_real.pdf"
    fake_pdf.write_bytes(b"this is not a valid pdf file")

    with pytest.raises(ProcessingError):
        parser.extract_pages(str(fake_pdf))


def test_parser_extracts_text_from_generated_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from page one.")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    pages = parser.extract_pages(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Hello" in pages[0]["text"]
