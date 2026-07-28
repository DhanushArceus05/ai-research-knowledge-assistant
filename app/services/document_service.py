"""
Document management service: upload validation, storage, listing, deletion,
reprocessing -- all scoped to the owning user.
"""
import os
import shutil
import uuid
import re
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import InvalidFileError, DocumentNotFoundError
from app.core.cache import get_cache_manager
from app.database.repositories import DocumentRepository, ChunkRepository
from app.database.models import ProcessingStatus, Document, User, UserRole
from app.document_processing.pipeline import DocumentPipeline
from app.vector_store.chroma_manager import get_chroma_manager
from app.core.logging import get_logger

logger = get_logger(__name__)

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(filename: str) -> str:
    name = os.path.basename(filename)
    return _FILENAME_SAFE_RE.sub("_", name)


class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.pipeline = DocumentPipeline()
        self.chroma = get_chroma_manager()
        self.cache = get_cache_manager()

    def _validate_file(self, file: UploadFile, size_bytes: int) -> None:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise InvalidFileError("Only PDF files are supported.")

        allowed_content_types = {"application/pdf", "application/octet-stream"}
        if file.content_type and file.content_type not in allowed_content_types:
            raise InvalidFileError(f"Unsupported content type: {file.content_type}")

        max_bytes = self.settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise InvalidFileError(
                f"File exceeds the maximum allowed size of {self.settings.MAX_UPLOAD_SIZE_MB} MB."
            )

    async def upload_document(self, file: UploadFile, user_id: str) -> Document:
        """Validates, stores, and creates an owned metadata record for one uploaded PDF."""
        content = await file.read()
        self._validate_file(file, len(content))

        document_id = str(uuid.uuid4())
        safe_name = _sanitize_filename(file.filename)
        stored_file_name = f"{document_id}_{safe_name}"
        file_path = os.path.join(self.settings.UPLOAD_DIR, stored_file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(content)
        except OSError as exc:
            # Clean up any partially written file on failure.
            if os.path.exists(file_path):
                os.remove(file_path)
            raise InvalidFileError(f"Failed to save the uploaded file: {exc}") from exc

        document = self.doc_repo.create(
            id=document_id,
            user_id=user_id,
            original_file_name=safe_name,
            stored_file_name=stored_file_name,
            file_path=file_path,
            processing_status=ProcessingStatus.PENDING,
        )
        return document

    def run_pipeline_sync(self, document_id: str, file_path: str, file_name: str, user_id: Optional[str]) -> None:
        """Runs the processing pipeline synchronously (used directly or via BackgroundTasks)."""
        self.pipeline.process(self.db, document_id, file_path, file_name, user_id)

    def list_documents(self, user_id: str) -> List[Document]:
        return self.doc_repo.list_all(user_id=user_id)

    def _is_owned_by(self, doc: Document, user: User) -> bool:
        """Ownership policy: a document is accessible to a user if they own it, or if the
        user is an admin. Legacy documents (user_id IS NULL, created before multi-user
        support existed) are NOT automatically exposed to any non-admin user."""
        if user.role == UserRole.ADMIN:
            return True
        return doc.user_id is not None and doc.user_id == user.id

    def get_owned_document(self, document_id: str, user: User) -> Document:
        """Fetches a document and enforces ownership. Raises 404 (not 403) for both
        'does not exist' and 'exists but not yours' to avoid leaking other users' document IDs."""
        doc = self.doc_repo.get(document_id)
        if not doc or not self._is_owned_by(doc, user):
            raise DocumentNotFoundError(document_id)
        return doc

    def get_document(self, document_id: str) -> Document:
        """Internal/no-ownership-check lookup, for background pipeline use only."""
        doc = self.doc_repo.get(document_id)
        if not doc:
            raise DocumentNotFoundError(document_id)
        return doc

    def _delete_asset_directories(self, document_id: str) -> None:
        images_dir = os.path.join(self.settings.IMAGES_DIR, document_id)
        if os.path.isdir(images_dir):
            shutil.rmtree(images_dir, ignore_errors=True)

    def delete_document(self, document_id: str, user: User) -> None:
        doc = self.get_owned_document(document_id, user)

        self.chunk_repo.delete_by_document(document_id)
        self.chroma.delete_document(document_id)
        self._delete_asset_directories(document_id)

        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except OSError:
                logger.warning("Could not remove file %s from disk.", doc.file_path)

        self.doc_repo.delete(document_id)
        self._invalidate_document_cache(document_id, doc.user_id)

    def reprocess_document(self, document_id: str, user: User) -> Document:
        doc = self.get_owned_document(document_id, user)
        if not os.path.exists(doc.file_path):
            raise InvalidFileError("The original file is no longer available on disk and cannot be reprocessed.")
        self._delete_asset_directories(document_id)
        self.pipeline.reprocess(self.db, doc.id, doc.file_path, doc.original_file_name, doc.user_id)
        self._invalidate_document_cache(document_id, doc.user_id)
        return self.get_owned_document(document_id, user)

    def _invalidate_document_cache(self, document_id: str, user_id: Optional[str]) -> None:
        """Clears any cached summaries/comparisons/search results that may reference this document."""
        owner = user_id or "system"
        for namespace in ("summary", "compare", "search", "doc_metadata", "analytics"):
            self.cache.delete_prefix(f"{namespace}:{owner}:")
