from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.database.repositories import DocumentRepository, ChunkRepository
from app.core.exceptions import DocumentNotFoundError, ProcessingError
from app.core.cache import get_cache_manager, make_cache_key
from app.rag.summarizer import SummarizationService
from app.rag.comparator import ComparisonService
from app.ml.predictor import get_document_classifier
from app.services.document_service import DocumentService
from app.schemas.analysis import (
    SummarizeRequest, SummarizeResponse,
    CompareRequest, CompareResponse,
    ClassifyRequest, ClassifyResponse,
)

router = APIRouter(prefix="/analysis", tags=["Document Analysis"])


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_document(payload: SummarizeRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Generates an Executive/Technical/Bullet-Point/Key-Takeaways summary for a document owned by the current user."""
    doc_service = DocumentService(db)
    chunk_repo = ChunkRepository(db)
    cache = get_cache_manager()

    document = doc_service.get_owned_document(payload.document_id, current_user)

    cache_key = make_cache_key("summary", current_user.id, document_id=document.id)
    cached = cache.get(cache_key)
    if cached is not None:
        return SummarizeResponse(document_id=document.id, file_name=document.original_file_name, summary=cached)

    chunks = chunk_repo.get_by_document(payload.document_id)
    if not chunks:
        raise ProcessingError("This document has no processed chunks yet. Wait for processing to complete.")

    service = SummarizationService()
    result = service.summarize_document(document.original_file_name, [c.text for c in chunks])

    cache.set(cache_key, result["summary"])

    return SummarizeResponse(
        document_id=document.id,
        file_name=document.original_file_name,
        summary=result["summary"],
    )


@router.post("/compare", response_model=CompareResponse)
def compare_documents(payload: CompareRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Compares two or more documents owned by the current user across methodology, similarities, differences, etc."""
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)
    cache = get_cache_manager()

    # Ownership-filtered fetch: any id not owned by the current user (or nonexistent) is
    # treated identically to "not found" so cross-user document IDs cannot be probed.
    documents = doc_repo.get_many(payload.document_ids, user_id=current_user.id)
    found_ids = {d.id for d in documents}
    missing = [d for d in payload.document_ids if d not in found_ids]
    if missing:
        raise DocumentNotFoundError(", ".join(missing))

    sorted_ids = sorted(d.id for d in documents)
    cache_key = make_cache_key("compare", current_user.id, document_ids=sorted_ids)
    cached = cache.get(cache_key)
    if cached is not None:
        return CompareResponse(
            document_ids=[d.id for d in documents],
            file_names=[d.original_file_name for d in documents],
            comparison=cached,
        )

    comparison_input = []
    for document in documents:
        chunks = chunk_repo.get_by_document(document.id)
        comparison_input.append({
            "file_name": document.original_file_name,
            "chunks": [{"page_number": c.page_number, "text": c.text} for c in chunks],
        })

    service = ComparisonService()
    comparison_text = service.compare_documents(comparison_input)

    cache.set(cache_key, comparison_text)

    return CompareResponse(
        document_ids=[d.id for d in documents],
        file_names=[d.original_file_name for d in documents],
        comparison=comparison_text,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_document(payload: ClassifyRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Runs (or re-runs) TensorFlow-based domain classification for a document owned by the current user."""
    doc_service = DocumentService(db)
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    document = doc_service.get_owned_document(payload.document_id, current_user)

    chunks = chunk_repo.get_by_document(payload.document_id)
    full_text = "\n".join(c.text for c in chunks)

    classifier = get_document_classifier()
    category, confidence = classifier.predict(full_text)

    doc_repo.update_classification(payload.document_id, category, confidence)

    return ClassifyResponse(document_id=document.id, predicted_category=category, confidence=confidence)
