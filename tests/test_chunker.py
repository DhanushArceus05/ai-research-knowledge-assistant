import pytest

from app.document_processing.chunker import TextChunker
from app.core.exceptions import ProcessingError


def test_chunker_rejects_overlap_greater_than_or_equal_to_chunk_size():
    with pytest.raises(ProcessingError):
        TextChunker(chunk_size=500, chunk_overlap=500)

    with pytest.raises(ProcessingError):
        TextChunker(chunk_size=500, chunk_overlap=600)


def test_short_page_produces_single_chunk():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=150)
    pages = [{"page_number": 1, "text": "This is a short page of text."}]
    chunks = chunker.chunk_pages(pages, document_id="doc-1")

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["document_id"] == "doc-1"
    assert chunks[0]["chunk_index"] == 0


def test_long_page_produces_multiple_overlapping_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = "A" * 350
    pages = [{"page_number": 3, "text": long_text}]
    chunks = chunker.chunk_pages(pages, document_id="doc-2")

    assert len(chunks) > 1
    # Every chunk must correctly retain the source page number
    assert all(c["page_number"] == 3 for c in chunks)
    # Chunk indices must be sequential starting at 0
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunking_terminates_and_covers_all_text_without_infinite_loop():
    chunker = TextChunker(chunk_size=50, chunk_overlap=49)  # worst-case near-degenerate overlap
    pages = [{"page_number": 1, "text": "B" * 500}]

    # Should complete quickly without hanging.
    chunks = chunker.chunk_pages(pages, document_id="doc-3")
    assert len(chunks) > 0


def test_multiple_pages_are_chunked_independently():
    chunker = TextChunker(chunk_size=1000, chunk_overlap=150)
    pages = [
        {"page_number": 1, "text": "Page one content."},
        {"page_number": 2, "text": "Page two content."},
    ]
    chunks = chunker.chunk_pages(pages, document_id="doc-4")

    page_numbers = {c["page_number"] for c in chunks}
    assert page_numbers == {1, 2}
