"""
Extracts tables from PDF pages using PyMuPDF's built-in structural table
finder (`page.find_tables()`, available in PyMuPDF >= 1.23). This avoids
adding an extra heavyweight dependency (e.g. pdfplumber/camelot) purely for
table extraction while still providing real, structural table detection
rather than a hard-coded/fake response.
"""
from typing import List, Dict, Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class TableExtractor:
    def extract_tables(self, doc) -> List[Dict[str, Any]]:
        """
        `doc` is an already-open `fitz.Document`. Returns a list of dicts ready to
        become ExtractedTable rows (without user_id/document_id, filled in by caller).
        """
        results: List[Dict[str, Any]] = []

        for page_index in range(doc.page_count):
            page = doc[page_index]
            try:
                table_finder = page.find_tables()
            except Exception:
                logger.warning("Table detection failed on page %d.", page_index + 1, exc_info=True)
                continue

            for table in table_finder.tables:
                try:
                    rows = table.extract()
                except Exception:
                    logger.warning("Failed to extract a detected table on page %d.", page_index + 1)
                    continue

                if not rows:
                    continue

                markdown = self._to_markdown(rows)
                results.append({
                    "page_number": page_index + 1,
                    "row_count": len(rows),
                    "column_count": len(rows[0]) if rows else 0,
                    "markdown": markdown,
                    "extraction_method": "pymupdf",
                    "confidence": 1.0,  # PyMuPDF does not expose a numeric confidence; structural detection = 1.0
                })

        return results

    def _to_markdown(self, rows: List[List[Any]]) -> str:
        def clean_cell(cell: Any) -> str:
            return str(cell).replace("\n", " ").strip() if cell is not None else ""

        if not rows:
            return ""

        header = rows[0]
        lines = ["| " + " | ".join(clean_cell(c) for c in header) + " |"]
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(clean_cell(c) for c in row) + " |")
        return "\n".join(lines)
