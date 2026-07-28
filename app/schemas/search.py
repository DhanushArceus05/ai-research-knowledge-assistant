from typing import Optional, List
from pydantic import BaseModel, field_validator


class SearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


class SearchResultItem(BaseModel):
    document_id: str
    file_name: str
    page_number: int
    text: str
    similarity: Optional[float] = None
    match_type: str


class SearchResponse(BaseModel):
    success: bool = True
    query: str
    result_count: int
    results: List[SearchResultItem]


class HybridSearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = None
    dense_candidates: Optional[int] = None
    sparse_candidates: Optional[int] = None
    apply_reranking: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("top_k", "dense_candidates", "sparse_candidates")
    @classmethod
    def must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("value must be a positive integer")
        return v


class HybridSearchResultItem(BaseModel):
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str
    fused_score: float
    semantic_score: Optional[float] = None
    semantic_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    match_type: str = "hybrid"


class HybridSearchResponse(BaseModel):
    success: bool = True
    query: str
    result_count: int
    results: List[HybridSearchResultItem]
    reranking_applied: bool
    reranking_note: str
