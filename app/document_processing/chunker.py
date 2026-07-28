"""
Page-aware overlapping text chunking.

Chunking strategy (documented further in README):
- Chunks are created independently per page so every chunk maps to exactly one
  page number, which keeps citations accurate.
- Default chunk size ~1000 characters with ~150 character overlap so that
  context near chunk boundaries is not lost during retrieval.
"""
from typing import List, Dict, Any

from app.core.exceptions import ProcessingError


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ProcessingError("chunk_overlap must be smaller than chunk_size to avoid an infinite loop.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]], document_id: str) -> List[Dict[str, Any]]:
        """
        Splits each page's text into overlapping chunks.
        Returns a list of dicts with: document_id, chunk_index, page_number, text.
        """
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap  # guaranteed > 0

        for page in pages:
            text = page["text"]
            page_number = page["page_number"]

            if len(text) <= self.chunk_size:
                chunks.append({
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "text": text,
                })
                chunk_index += 1
                continue

            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "page_number": page_number,
                        "text": chunk_text,
                    })
                    chunk_index += 1
                if end >= text_len:
                    break
                start += step

        return chunks
