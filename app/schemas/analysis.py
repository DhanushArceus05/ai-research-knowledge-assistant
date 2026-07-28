from typing import List
from pydantic import BaseModel, field_validator


class SummarizeRequest(BaseModel):
    document_id: str

    @field_validator("document_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id must not be empty")
        return v


class SummarizeResponse(BaseModel):
    success: bool = True
    document_id: str
    file_name: str
    summary: str


class CompareRequest(BaseModel):
    document_ids: List[str]

    @field_validator("document_ids")
    @classmethod
    def at_least_two(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 2:
            raise ValueError("At least two document_ids are required for comparison")
        return v


class CompareResponse(BaseModel):
    success: bool = True
    document_ids: List[str]
    file_names: List[str]
    comparison: str


class ClassifyRequest(BaseModel):
    document_id: str

    @field_validator("document_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id must not be empty")
        return v


class ClassifyResponse(BaseModel):
    success: bool = True
    document_id: str
    predicted_category: str
    confidence: float
