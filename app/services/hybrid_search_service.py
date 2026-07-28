"""
Hybrid retrieval: combines dense vector similarity (ChromaDB) with sparse
BM25 keyword relevance, fusing the two ranked candidate lists with
Reciprocal Rank Fusion (RRF).

RRF is used (rather than a weighted linear combination of raw scores)
because BM25 scores and cosine-similarity scores live on entirely different,
non-comparable scales; RRF sidesteps that by fusing on RANK rather than raw
score, which is simple, robust, and requires no tuning of relative weights.

    RRF(d) = sum over each ranked list L containing d of  1 / (k + rank_L(d))

A small constant `k` (default 60, the value from the original RRF paper)
dampens the influence of very high ranks so the fusion isn't dominated by a
single list.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.repositories import ChunkRepository, DocumentRepository
from app.vector_store.chroma_manager import get_chroma_manager
from app.vector_store.bm25_index import build_bm25_index
from app.core.logging import get_logger

logger = get_logger(__name__)


def _fusion_key(document_id: str, chunk_index: Any) -> str:
    return f"{document_id}:{chunk_index}"


class HybridSearchService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.chunk_repo = ChunkRepository(db)
        self.doc_repo = DocumentRepository(db)
        self.chroma = get_chroma_manager()

    def search(
        self,
        query: str,
        user_id: Optional[str],
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        dense_candidates: Optional[int] = None,
        sparse_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.settings.TOP_K_RESULTS
        n_dense = dense_candidates or self.settings.HYBRID_DENSE_CANDIDATES
        n_sparse = sparse_candidates or self.settings.HYBRID_SPARSE_CANDIDATES
        rrf_k = self.settings.RRF_K

        dense_results = self.chroma.semantic_search(
            query, top_k=n_dense, document_ids=document_ids, user_id=user_id
        )
        bm25_index = build_bm25_index(self.chunk_repo, user_id=user_id, document_ids=document_ids)
        sparse_results = bm25_index.search(query, top_k=n_sparse)

        # Build a fused record per unique chunk, tracking rank/score from each source list.
        fused: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            key = _fusion_key(item["document_id"], item["chunk_index"])
            fused[key] = {
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "page_number": item["page_number"],
                "chunk_index": item["chunk_index"],
                "text": item["text"],
                "semantic_score": item["similarity"],
                "semantic_rank": rank,
                "bm25_score": None,
                "bm25_rank": None,
            }

        # Resolve file names for BM25-only chunks (BM25 index doesn't carry file_name).
        doc_name_cache: Dict[str, str] = {}

        for rank, item in enumerate(sparse_results, start=1):
            key = _fusion_key(item["document_id"], item["chunk_index"])
            if key in fused:
                fused[key]["bm25_score"] = item["bm25_score"]
                fused[key]["bm25_rank"] = rank
            else:
                if item["document_id"] not in doc_name_cache:
                    doc = self.doc_repo.get(item["document_id"])
                    doc_name_cache[item["document_id"]] = doc.original_file_name if doc else "Unknown"
                fused[key] = {
                    "document_id": item["document_id"],
                    "file_name": doc_name_cache[item["document_id"]],
                    "page_number": item["page_number"],
                    "chunk_index": item["chunk_index"],
                    "text": item["text"],
                    "semantic_score": None,
                    "semantic_rank": None,
                    "bm25_score": item["bm25_score"],
                    "bm25_rank": rank,
                }

        # Reciprocal Rank Fusion score.
        for record in fused.values():
            score = 0.0
            if record["semantic_rank"] is not None:
                score += 1.0 / (rrf_k + record["semantic_rank"])
            if record["bm25_rank"] is not None:
                score += 1.0 / (rrf_k + record["bm25_rank"])
            record["fused_score"] = score

        ranked = sorted(fused.values(), key=lambda r: r["fused_score"], reverse=True)
        top_results = ranked[:k]

        for r in top_results:
            r["match_type"] = "hybrid"

        return top_results
