from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.services.search_service import SearchService
from app.services.hybrid_search_service import HybridSearchService
from app.vector_store.reranker import get_reranker
from app.core.config import get_settings
from app.core.cache import get_cache_manager, make_cache_key
from app.schemas.search import (
    SearchRequest, SearchResponse, SearchResultItem,
    HybridSearchRequest, HybridSearchResponse, HybridSearchResultItem,
)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/keyword", response_model=SearchResponse)
def keyword_search(payload: SearchRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Case-insensitive exact-text search over the current user's indexed document chunks."""
    service = SearchService(db)
    results = service.keyword_search(payload.query, user_id=current_user.id, document_ids=payload.document_ids)
    return SearchResponse(
        query=payload.query,
        result_count=len(results),
        results=[SearchResultItem(**r) for r in results],
    )


@router.post("/semantic", response_model=SearchResponse)
def semantic_search(payload: SearchRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Meaning-based vector similarity search over the current user's indexed document chunks."""
    service = SearchService(db)
    results = service.semantic_search(
        payload.query, user_id=current_user.id, document_ids=payload.document_ids, top_k=payload.top_k
    )
    return SearchResponse(
        query=payload.query,
        result_count=len(results),
        results=[SearchResultItem(**r) for r in results],
    )


@router.post("/hybrid", response_model=HybridSearchResponse)
def hybrid_search(payload: HybridSearchRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Hybrid retrieval combining dense vector similarity (ChromaDB) with sparse BM25
    keyword relevance, fused via Reciprocal Rank Fusion, with an optional second-stage
    cross-encoder reranking pass (falls back gracefully if the reranker is unavailable).
    Results are cached per-user for a short TTL since retrieval for the same query
    over an unchanged document set is deterministic-ish and comparatively expensive.
    """
    settings = get_settings()
    cache = get_cache_manager()

    cache_key = make_cache_key(
        "search", current_user.id,
        query=payload.query, document_ids=sorted(payload.document_ids or []),
        top_k=payload.top_k, apply_reranking=payload.apply_reranking,
        dense_candidates=payload.dense_candidates, sparse_candidates=payload.sparse_candidates,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return HybridSearchResponse(**cached)

    service = HybridSearchService(db)

    candidates = service.search(
        query=payload.query,
        user_id=current_user.id,
        top_k=(payload.top_k or settings.TOP_K_RESULTS) * 2 if payload.apply_reranking else payload.top_k,
        document_ids=payload.document_ids,
        dense_candidates=payload.dense_candidates,
        sparse_candidates=payload.sparse_candidates,
    )

    final_top_k = payload.top_k or settings.TOP_K_RESULTS
    reranking_applied = False
    reranking_note = "Reranking was not requested for this search."

    if payload.apply_reranking:
        reranker = get_reranker()
        candidates, reranking_applied, reranking_note = reranker.rerank(payload.query, candidates, final_top_k)
    else:
        candidates = candidates[:final_top_k]

    response = HybridSearchResponse(
        query=payload.query,
        result_count=len(candidates),
        results=[HybridSearchResultItem(**c) for c in candidates],
        reranking_applied=reranking_applied,
        reranking_note=reranking_note,
    )
    cache.set(cache_key, response.model_dump())
    return response
