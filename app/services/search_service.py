"""
Search service supporting keyword and semantic search modes, both scoped to
the requesting user's own documents.

Keyword search is best for exact terms, names, IDs, or phrases that must match
literally. Semantic search is best for meaning-based, contextual questions
where the exact wording may differ from the source text. See
HybridSearchService for the combined BM25 + vector retrieval mode.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.repositories import ChunkRepository, DocumentRepository
from app.vector_store.chroma_manager import get_chroma_manager


class SearchService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.chunk_repo = ChunkRepository(db)
        self.doc_repo = DocumentRepository(db)
        self.chroma = get_chroma_manager()

    def keyword_search(
        self, query: str, user_id: Optional[str], document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        chunks = self.chunk_repo.keyword_search(query, document_ids=document_ids, user_id=user_id)
        results = []
        for c in chunks:
            doc = self.doc_repo.get(c.document_id)
            results.append({
                "document_id": c.document_id,
                "file_name": doc.original_file_name if doc else "Unknown",
                "page_number": c.page_number,
                "text": c.text,
                "similarity": None,
                "match_type": "keyword",
            })
        return results

    def semantic_search(
        self,
        query: str,
        user_id: Optional[str],
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.settings.TOP_K_RESULTS
        raw_results = self.chroma.semantic_search(query, top_k=k, document_ids=document_ids, user_id=user_id)
        results = []
        for r in raw_results:
            results.append({
                "document_id": r["document_id"],
                "file_name": r["file_name"],
                "page_number": r["page_number"],
                "text": r["text"],
                "similarity": r["similarity"],
                "match_type": "semantic",
            })
        return results
