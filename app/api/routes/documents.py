from typing import List

from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.database.repositories import ImageAssetRepository, ExtractedTableRepository
from app.database.models import ProcessingStatus
from app.services.document_service import DocumentService
from app.core.cache import get_cache_manager, make_cache_key
from app.schemas.documents import (
    DocumentUploadResponse,
    DocumentUploadResult,
    DocumentListResponse,
    DocumentResponse,
    DocumentDeleteResponse,
    DocumentReprocessResponse,
    ImageListResponse,
    ImageAssetResponse,
    TableListResponse,
    ExtractedTableResponse,
)

router = APIRouter(prefix="/documents", tags=["Document Management"])


def _run_pipeline_in_background(document_id: str, file_path: str, file_name: str, user_id: str):
    """Standalone background task target: opens its own DB session to stay thread-safe."""
    from app.database.session import get_db as _get_db
    db_gen = _get_db()
    db = next(db_gen)
    try:
        service = DocumentService(db)
        service.run_pipeline_sync(document_id, file_path, file_name, user_id)
    finally:
        db.close()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Uploads one or more PDF documents (owned by the current user) and triggers background processing for each."""
    service = DocumentService(db)
    results = []

    for file in files:
        document = await service.upload_document(file, current_user.id)
        background_tasks.add_task(
            _run_pipeline_in_background, document.id, document.file_path, document.original_file_name, current_user.id
        )
        results.append(DocumentUploadResult(
            document_id=document.id,
            file_name=document.original_file_name,
            processing_status=document.processing_status.value,
            message="Document uploaded successfully. Processing started in the background.",
        ))

    return DocumentUploadResponse(results=results)


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Lists all documents owned by the current user."""
    service = DocumentService(db)
    documents = service.list_documents(current_user.id)
    return DocumentListResponse(
        count=len(documents),
        documents=[DocumentResponse(**d.to_dict()) for d in documents],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retrieves metadata for a single document owned by the current user.
    Cached only once processing has reached a terminal state (PROCESSED/FAILED),
    so an in-progress document's status is never served stale."""
    cache = get_cache_manager()
    cache_key = make_cache_key("doc_metadata", current_user.id, document_id=document_id)
    cached = cache.get(cache_key)
    if cached is not None:
        return DocumentResponse(**cached)

    service = DocumentService(db)
    document = service.get_owned_document(document_id, current_user)
    response = DocumentResponse(**document.to_dict())

    if document.processing_status in (ProcessingStatus.PROCESSED, ProcessingStatus.FAILED):
        cache.set(cache_key, response.model_dump())

    return response


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Deletes a document owned by the current user, its chunks, vectors, extracted assets, and stored file."""
    service = DocumentService(db)
    service.delete_document(document_id, current_user)
    return DocumentDeleteResponse(message=f"Document '{document_id}' deleted successfully.")


@router.post("/{document_id}/reprocess", response_model=DocumentReprocessResponse)
def reprocess_document(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Reprocesses a document owned by the current user from scratch."""
    service = DocumentService(db)
    document = service.reprocess_document(document_id, current_user)
    return DocumentReprocessResponse(
        document_id=document.id,
        processing_status=document.processing_status.value,
        message="Document reprocessed successfully.",
    )


@router.get("/{document_id}/images", response_model=ImageListResponse)
def get_document_images(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Lists metadata for images embedded in this document's source pages. Ownership is enforced."""
    service = DocumentService(db)
    service.get_owned_document(document_id, current_user)  # ownership + existence check
    images = ImageAssetRepository(db).get_by_document(document_id)
    return ImageListResponse(
        document_id=document_id,
        count=len(images),
        images=[ImageAssetResponse(**img.to_dict()) for img in images],
    )


@router.get("/{document_id}/tables", response_model=TableListResponse)
def get_document_tables(document_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Lists tables extracted from this document's pages. Ownership is enforced."""
    service = DocumentService(db)
    service.get_owned_document(document_id, current_user)  # ownership + existence check
    tables = ExtractedTableRepository(db).get_by_document(document_id)
    return TableListResponse(
        document_id=document_id,
        count=len(tables),
        tables=[ExtractedTableResponse(**t.to_dict()) for t in tables],
    )
