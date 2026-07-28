"""
Analytics aggregation service. By default scoped to a single user; pass
user_id=None (admin-only endpoint) for system-wide analytics.
"""
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.database.repositories import DocumentRepository, ChunkRepository, QueryLogRepository
from app.database.models import ProcessingStatus
from app.vector_store.chroma_manager import get_chroma_manager


class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.query_log_repo = QueryLogRepository(db)
        self.chroma = get_chroma_manager()

    def get_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        category_distribution = {
            category or "UNCLASSIFIED": count
            for category, count in self.doc_repo.category_distribution(user_id=user_id)
        }

        most_queried = [
            {"document_id": d.id, "file_name": d.original_file_name, "query_count": d.query_count}
            for d in self.doc_repo.most_queried(limit=5, user_id=user_id)
        ]

        recent_queries = [
            {"question": q.question, "created_at": q.created_at.isoformat()}
            for q in self.query_log_repo.recent(limit=10, user_id=user_id)
        ]

        return {
            "total_documents": self.doc_repo.total_count(user_id=user_id),
            "total_processed_documents": self.doc_repo.count_by_status(ProcessingStatus.PROCESSED, user_id=user_id),
            "total_failed_documents": self.doc_repo.count_by_status(ProcessingStatus.FAILED, user_id=user_id),
            "total_chunks": self.chunk_repo.total_count(user_id=user_id),
            "total_embeddings": self.chroma.count(user_id=user_id),
            "total_questions_answered": self.query_log_repo.total_count(user_id=user_id),
            "category_distribution": category_distribution,
            "most_queried_documents": most_queried,
            "recent_queries": recent_queries,
        }
