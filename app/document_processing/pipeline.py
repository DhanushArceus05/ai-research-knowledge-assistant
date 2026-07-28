"""
Orchestrates the full per-document processing pipeline:
parse -> OCR fallback -> clean -> chunk -> classify -> embed -> index ->
extract images/tables -> persist metadata.

The source PDF is opened exactly once via PyMuPDF and that single open
document is reused across text extraction, OCR fallback, image extraction,
and table extraction, then closed at the end of the run.
"""
from typing import Optional
from sqlalchemy.orm import Session

from app.document_processing.parser import PDFParser
from app.document_processing.cleaner import TextCleaner
from app.document_processing.chunker import TextChunker
from app.document_processing.ocr import get_ocr_service
from app.document_processing.image_extractor import ImageExtractor
from app.document_processing.table_extractor import TableExtractor
from app.vector_store.chroma_manager import get_chroma_manager
from app.ml.predictor import get_document_classifier
from app.database.repositories import (
    DocumentRepository, ChunkRepository, ImageAssetRepository, ExtractedTableRepository,
)
from app.database.models import ProcessingStatus
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.cache import get_cache_manager

logger = get_logger(__name__)


class DocumentPipeline:
    def __init__(self):
        settings = get_settings()
        self.parser = PDFParser()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        self.chroma = get_chroma_manager()
        self.classifier = get_document_classifier()
        self.ocr = get_ocr_service()
        self.image_extractor = ImageExtractor()
        self.table_extractor = TableExtractor()

    def _extract_pages_with_ocr_fallback(self, doc):
        """Extracts text per page; for pages with too little extractable text, attempts
        OCR (if Tesseract is available) and reports which method produced the final text."""
        pages = self.parser.extract_pages_from_doc(doc)
        ocr_pages_attempted = 0
        ocr_pages_succeeded = 0

        for page in pages:
            if self.ocr.needs_ocr(page["text"]):
                ocr_pages_attempted += 1
                ocr_text = self.ocr.extract_text_from_page_image(doc[page["page_number"] - 1])
                if ocr_text:
                    page["text"] = ocr_text
                    page["extraction_method"] = "ocr"
                    ocr_pages_succeeded += 1
                # If OCR is unavailable/failed, the page keeps its (possibly empty) text
                # extraction_method stays "text"; the page is simply dropped later if empty.

        if ocr_pages_attempted and not self.ocr.is_available():
            logger.warning(
                "%d page(s) had sparse text but OCR is unavailable; content may be missing.",
                ocr_pages_attempted,
            )
        elif ocr_pages_attempted:
            logger.info("OCR attempted on %d page(s), succeeded on %d.", ocr_pages_attempted, ocr_pages_succeeded)

        # Drop pages that ended up with no usable text at all (even after OCR attempt).
        return [p for p in pages if p["text"].strip()]

    def process(self, db: Session, document_id: str, file_path: str, file_name: str, user_id: Optional[str] = None) -> None:
        """Runs the full pipeline for a single document, updating its DB status as it progresses."""
        doc_repo = DocumentRepository(db)
        chunk_repo = ChunkRepository(db)
        image_repo = ImageAssetRepository(db)
        table_repo = ExtractedTableRepository(db)

        try:
            doc_repo.update_status(document_id, ProcessingStatus.PROCESSING)

            fitz_doc = self.parser.open_document(file_path)
            try:
                pages = self._extract_pages_with_ocr_fallback(fitz_doc)
                if not pages:
                    raise ValueError(
                        "No extractable text was found in this PDF, even after attempting OCR. "
                        "It may be a scanned document and OCR (Tesseract) may not be installed."
                    )

                for page in pages:
                    page["text"] = self.cleaner.clean(page["text"])

                chunks = self.chunker.chunk_pages(pages, document_id)
                if not chunks:
                    raise ValueError("No chunks were produced from this document.")

                # Attach extraction_method + user_id to each chunk for persistence.
                method_by_page = {p["page_number"]: p["extraction_method"] for p in pages}
                for chunk in chunks:
                    chunk["extraction_method"] = method_by_page.get(chunk["page_number"], "text")
                    chunk["user_id"] = user_id

                full_text = "\n".join(p["text"] for p in pages)
                category, confidence = self.classifier.predict(full_text)

                self.chroma.index_chunks(chunks, file_name, user_id=user_id)
                chunk_repo.bulk_create(chunks)

                # Multi-modal extraction: embedded images + structural tables.
                try:
                    images = self.image_extractor.extract_images(fitz_doc, document_id)
                    for img in images:
                        img["document_id"] = document_id
                        img["user_id"] = user_id
                    image_repo.bulk_create(images)
                except Exception:
                    logger.warning("Image extraction failed for document %s; continuing without it.", document_id, exc_info=True)

                try:
                    tables = self.table_extractor.extract_tables(fitz_doc)
                    for tbl in tables:
                        tbl["document_id"] = document_id
                        tbl["user_id"] = user_id
                    table_repo.bulk_create(tables)
                except Exception:
                    logger.warning("Table extraction failed for document %s; continuing without it.", document_id, exc_info=True)
            finally:
                fitz_doc.close()

            doc_repo.update_pipeline_result(document_id, total_pages=len(pages), total_chunks=len(chunks))
            doc_repo.update_classification(document_id, category, confidence)
            doc_repo.update_status(document_id, ProcessingStatus.PROCESSED)

            logger.info(
                "Document %s processed successfully: %d pages, %d chunks, category=%s",
                document_id, len(pages), len(chunks), category,
            )
        except Exception as exc:
            logger.exception("Processing failed for document %s", document_id)
            doc_repo.update_status(document_id, ProcessingStatus.FAILED, error_message=str(exc))
        finally:
            owner = user_id or "system"
            cache = get_cache_manager()
            for namespace in ("analytics", "search", "doc_metadata"):
                cache.delete_prefix(f"{namespace}:{owner}:")

    def reprocess(self, db: Session, document_id: str, file_path: str, file_name: str, user_id: Optional[str] = None) -> None:
        """Removes existing chunks/vectors/images/tables for a document before reprocessing it from scratch."""
        chunk_repo = ChunkRepository(db)
        image_repo = ImageAssetRepository(db)
        table_repo = ExtractedTableRepository(db)

        chunk_repo.delete_by_document(document_id)
        image_repo.delete_by_document(document_id)
        table_repo.delete_by_document(document_id)
        self.chroma.delete_document(document_id)
        self.process(db, document_id, file_path, file_name, user_id)
