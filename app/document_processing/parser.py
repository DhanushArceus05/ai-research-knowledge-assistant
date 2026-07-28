"""
PDF text extraction using PyMuPDF (fitz), preserving one-based page numbers.
"""
from typing import List, Dict, Any

import fitz  # PyMuPDF

from app.core.exceptions import ProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Extracts page-level text and metadata from a PDF file."""

    def open_document(self, file_path: str):
        """Opens a PDF and returns the fitz.Document. Caller is responsible for closing it."""
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise ProcessingError(f"Unable to open PDF file: {exc}") from exc

        if doc.page_count == 0:
            doc.close()
            raise ProcessingError("The PDF file contains no pages.")
        return doc

    def extract_pages_from_doc(self, doc) -> List[Dict[str, Any]]:
        """Extracts page-level text (including empty pages, unlike extract_pages) from an
        already-open fitz.Document, so the pipeline can also OCR pages with sparse text."""
        pages: List[Dict[str, Any]] = []
        for page_index in range(doc.page_count):
            page = doc[page_index]
            text = (page.get_text("text") or "").strip()
            pages.append({"page_number": page_index + 1, "text": text, "extraction_method": "text"})
        return pages

    def extract_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of {"page_number": int (1-based), "text": str} for every
        non-empty page in the PDF. Raises ProcessingError for damaged/invalid files.
        """
        doc = self.open_document(file_path)
        try:
            pages = [p for p in self.extract_pages_from_doc(doc) if p["text"]]
        finally:
            doc.close()

        if not pages:
            raise ProcessingError("No extractable text was found in the PDF (it may be a scanned/image-only file).")

        return pages

    def get_page_count(self, file_path: str) -> int:
        try:
            doc = fitz.open(file_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as exc:
            raise ProcessingError(f"Unable to read PDF page count: {exc}") from exc
