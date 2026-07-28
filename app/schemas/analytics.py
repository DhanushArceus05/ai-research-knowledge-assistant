from typing import List, Dict
from pydantic import BaseModel


class MostQueriedDocument(BaseModel):
    document_id: str
    file_name: str
    query_count: int


class RecentQuery(BaseModel):
    question: str
    created_at: str


class AnalyticsResponse(BaseModel):
    success: bool = True
    total_documents: int
    total_processed_documents: int
    total_failed_documents: int
    total_chunks: int
    total_embeddings: int
    total_questions_answered: int
    category_distribution: Dict[str, int]
    most_queried_documents: List[MostQueriedDocument]
    recent_queries: List[RecentQuery]
