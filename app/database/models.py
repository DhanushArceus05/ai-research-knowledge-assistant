"""
SQLAlchemy ORM models: User, Document, Chunk, ConversationSession, ConversationMessage,
QueryLog, ImageAsset, ExtractedTable.
"""
import uuid
import enum
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean,
    ForeignKey, Enum as SAEnum, Index,
)
from sqlalchemy.orm import relationship

from app.database.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ProcessingStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    role = Column(SAEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "user_id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role.value if self.role else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)

    # Nullable for backward compatibility with rows created before multi-user
    # support was added. See scripts/migrate_db.py / README migration notes.
    # New documents always have an owner (enforced at the service layer).
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)

    original_file_name = Column(String, nullable=False)
    stored_file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)

    processing_status = Column(SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)

    predicted_category = Column(String, default="UNCLASSIFIED")
    classification_confidence = Column(Float, default=0.0)

    query_count = Column(Integer, default=0)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    images = relationship("ImageAsset", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("ExtractedTable", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "document_id": self.id,
            "user_id": self.user_id,
            "file_name": self.original_file_name,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_status": self.processing_status.value if self.processing_status else None,
            "error_message": self.error_message,
            "predicted_category": self.predicted_category,
            "classification_confidence": self.classification_confidence,
            "query_count": self.query_count,
        }


class Chunk(Base):
    """
    Stores chunk text redundantly in SQLite (in addition to ChromaDB) so that
    keyword/BM25 search and summarization/comparison can operate without
    depending on the vector store's internal representation.
    """

    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    extraction_method = Column(String, default="text")  # "text" or "ocr"

    document = relationship("Document", back_populates="chunks")


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    messages = relationship(
        "ConversationMessage", back_populates="session", cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("conversation_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ConversationSession", back_populates="messages")


class QueryLog(Base):
    """Records each question answered for analytics purposes."""

    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String, nullable=True)
    question = Column(Text, nullable=False)
    document_ids = Column(Text, nullable=True)  # comma-separated
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ImageAsset(Base):
    """Metadata for an image embedded in a source PDF page."""

    __tablename__ = "image_assets"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    page_number = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String, nullable=True)
    description = Column(Text, nullable=True)  # optional Gemini-vision caption
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="images")

    def to_dict(self) -> dict:
        return {
            "image_id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "description": self.description,
        }


class ExtractedTable(Base):
    """Metadata + content for a table extracted from a source PDF page."""

    __tablename__ = "extracted_tables"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    page_number = Column(Integer, nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    markdown = Column(Text, nullable=True)
    extraction_method = Column(String, default="pymupdf")
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="tables")

    def to_dict(self) -> dict:
        return {
            "table_id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "markdown": self.markdown,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
        }


# Composite indexes for common ownership + lookup query patterns.
Index("ix_documents_user_status", Document.user_id, Document.processing_status)
Index("ix_chunks_document_user", Chunk.document_id, Chunk.user_id)
