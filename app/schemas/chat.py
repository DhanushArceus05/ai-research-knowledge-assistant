from typing import Optional, List, Any
from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = None
    retrieval_mode: str = "semantic"
    apply_reranking: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question must not be empty")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @field_validator("retrieval_mode")
    @classmethod
    def retrieval_mode_must_be_valid(cls, v: str) -> str:
        if v not in ("semantic", "keyword", "hybrid"):
            raise ValueError("retrieval_mode must be one of: semantic, keyword, hybrid")
        return v


class Citation(BaseModel):
    document: str
    page: Any


class RetrievedContextItem(BaseModel):
    document_id: str
    file_name: str
    page_number: Any
    text: str
    similarity: Optional[float] = None


class ChatResponse(BaseModel):
    success: bool = True
    answer: str
    citations: List[Citation]
    source_documents: List[str]
    retrieved_context: List[RetrievedContextItem]
    session_id: str
    confidence: Optional[float] = None
    retrieval_mode: str = "semantic"


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: str


class SessionHistoryResponse(BaseModel):
    success: bool = True
    session_id: str
    messages: List[SessionMessage]


class SessionDeleteResponse(BaseModel):
    success: bool = True
    message: str
