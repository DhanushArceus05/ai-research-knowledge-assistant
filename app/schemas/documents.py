from typing import Optional, List
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: str
    file_name: str
    upload_timestamp: Optional[str] = None
    total_pages: int
    total_chunks: int
    processing_status: str
    error_message: Optional[str] = None
    predicted_category: str
    classification_confidence: float
    query_count: int


class DocumentUploadResult(BaseModel):
    document_id: str
    file_name: str
    processing_status: str
    message: str


class DocumentUploadResponse(BaseModel):
    success: bool = True
    results: List[DocumentUploadResult]


class DocumentListResponse(BaseModel):
    success: bool = True
    count: int
    documents: List[DocumentResponse]


class DocumentDeleteResponse(BaseModel):
    success: bool = True
    message: str


class DocumentReprocessResponse(BaseModel):
    success: bool = True
    document_id: str
    processing_status: str
    message: str


class ImageAssetResponse(BaseModel):
    image_id: str
    document_id: str
    page_number: int
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    description: Optional[str] = None


class ImageListResponse(BaseModel):
    success: bool = True
    document_id: str
    count: int
    images: List[ImageAssetResponse]


class ExtractedTableResponse(BaseModel):
    table_id: str
    document_id: str
    page_number: int
    row_count: int
    column_count: int
    markdown: Optional[str] = None
    extraction_method: str
    confidence: float


class TableListResponse(BaseModel):
    success: bool = True
    document_id: str
    count: int
    tables: List[ExtractedTableResponse]
