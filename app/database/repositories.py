"""
Repository layer: encapsulates all direct SQLAlchemy queries used by services.

Ownership filtering: most read/aggregate methods accept an optional
`user_id` parameter. When provided, results are scoped to that user only.
When `user_id` is omitted entirely, the method returns system-wide results;
this is only used internally (e.g. by the background processing pipeline,
which operates on a document it already knows the id of) or by admin-only
endpoints.
"""
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import (
    User,
    Document,
    Chunk,
    ProcessingStatus,
    ConversationSession,
    ConversationMessage,
    QueryLog,
    ImageAsset,
    ExtractedTable,
)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, email: str, hashed_password: str, display_name: Optional[str], role) -> User:
        user = User(email=email, hashed_password=hashed_password, display_name=display_name, role=role)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **kwargs) -> Document:
        doc = Document(**kwargs)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get(self, document_id: str) -> Optional[Document]:
        """Fetches a document by id with NO ownership filtering. Callers performing
        a user-facing request must separately verify ownership before returning
        data (see DocumentService.get_owned_document)."""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_all(self, user_id: Optional[str] = None) -> List[Document]:
        query = self.db.query(Document)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.order_by(Document.upload_timestamp.desc()).all()

    def get_many(self, document_ids: List[str], user_id: Optional[str] = None) -> List[Document]:
        query = self.db.query(Document).filter(Document.id.in_(document_ids))
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.all()

    def update_status(self, document_id: str, status: ProcessingStatus, error_message: Optional[str] = None) -> None:
        doc = self.get(document_id)
        if doc:
            doc.processing_status = status
            doc.error_message = error_message
            self.db.commit()

    def update_classification(self, document_id: str, category: str, confidence: float) -> None:
        doc = self.get(document_id)
        if doc:
            doc.predicted_category = category
            doc.classification_confidence = confidence
            self.db.commit()

    def update_pipeline_result(self, document_id: str, total_pages: int, total_chunks: int) -> None:
        doc = self.get(document_id)
        if doc:
            doc.total_pages = total_pages
            doc.total_chunks = total_chunks
            self.db.commit()

    def increment_query_count(self, document_id: str) -> None:
        doc = self.get(document_id)
        if doc:
            doc.query_count = (doc.query_count or 0) + 1
            self.db.commit()

    def delete(self, document_id: str) -> None:
        doc = self.get(document_id)
        if doc:
            self.db.delete(doc)
            self.db.commit()

    def count_by_status(self, status: ProcessingStatus, user_id: Optional[str] = None) -> int:
        query = self.db.query(Document).filter(Document.processing_status == status)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.count()

    def total_count(self, user_id: Optional[str] = None) -> int:
        query = self.db.query(Document)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.count()

    def category_distribution(self, user_id: Optional[str] = None) -> List[tuple]:
        query = self.db.query(Document.predicted_category, func.count(Document.id))
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.group_by(Document.predicted_category).all()

    def most_queried(self, limit: int = 5, user_id: Optional[str] = None) -> List[Document]:
        query = self.db.query(Document).filter(Document.query_count > 0)
        if user_id is not None:
            query = query.filter(Document.user_id == user_id)
        return query.order_by(Document.query_count.desc()).limit(limit).all()


class ChunkRepository:
    def __init__(self, db: Session):
        self.db = db

    def bulk_create(self, chunks: List[dict]) -> None:
        objects = [Chunk(**c) for c in chunks]
        self.db.add_all(objects)
        self.db.commit()

    def get_by_document(self, document_id: str) -> List[Chunk]:
        return (
            self.db.query(Chunk)
            .filter(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
            .all()
        )

    def get_all_for_user(self, user_id: Optional[str] = None, document_ids: Optional[List[str]] = None) -> List[Chunk]:
        """Used by BM25 index building: returns every chunk owned by a user (or all chunks if user_id is None)."""
        query = self.db.query(Chunk)
        if user_id is not None:
            query = query.filter(Chunk.user_id == user_id)
        if document_ids:
            query = query.filter(Chunk.document_id.in_(document_ids))
        return query.all()

    def keyword_search(
        self, keyword: str, document_ids: Optional[List[str]] = None, user_id: Optional[str] = None, limit: int = 20
    ) -> List[Chunk]:
        query = self.db.query(Chunk).filter(Chunk.text.ilike(f"%{keyword}%"))
        if document_ids:
            query = query.filter(Chunk.document_id.in_(document_ids))
        if user_id is not None:
            query = query.filter(Chunk.user_id == user_id)
        return query.limit(limit).all()

    def delete_by_document(self, document_id: str) -> None:
        self.db.query(Chunk).filter(Chunk.document_id == document_id).delete()
        self.db.commit()

    def total_count(self, user_id: Optional[str] = None) -> int:
        query = self.db.query(Chunk)
        if user_id is not None:
            query = query.filter(Chunk.user_id == user_id)
        return query.count()


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_session(self, session_id: Optional[str], user_id: Optional[str] = None) -> ConversationSession:
        if session_id:
            existing = self.db.query(ConversationSession).filter(ConversationSession.id == session_id).first()
            if existing:
                return existing
        new_session = (
            ConversationSession(id=session_id, user_id=user_id)
            if session_id
            else ConversationSession(user_id=user_id)
        )
        self.db.add(new_session)
        self.db.commit()
        self.db.refresh(new_session)
        return new_session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self.db.query(ConversationSession).filter(ConversationSession.id == session_id).first()

    def add_message(self, session_id: str, role: str, content: str) -> ConversationMessage:
        message = ConversationMessage(session_id=session_id, role=role, content=content)
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_recent_messages(self, session_id: str, limit: int) -> List[ConversationMessage]:
        messages = (
            self.db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(messages))

    def clear_session(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        self.db.delete(session)
        self.db.commit()
        return True


class QueryLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def log(
        self,
        question: str,
        session_id: Optional[str],
        document_ids: Optional[List[str]],
        user_id: Optional[str] = None,
    ) -> None:
        entry = QueryLog(
            question=question,
            session_id=session_id,
            document_ids=",".join(document_ids) if document_ids else None,
            user_id=user_id,
        )
        self.db.add(entry)
        self.db.commit()

    def total_count(self, user_id: Optional[str] = None) -> int:
        query = self.db.query(QueryLog)
        if user_id is not None:
            query = query.filter(QueryLog.user_id == user_id)
        return query.count()

    def recent(self, limit: int = 10, user_id: Optional[str] = None) -> List[QueryLog]:
        query = self.db.query(QueryLog)
        if user_id is not None:
            query = query.filter(QueryLog.user_id == user_id)
        return query.order_by(QueryLog.created_at.desc()).limit(limit).all()


class ImageAssetRepository:
    def __init__(self, db: Session):
        self.db = db

    def bulk_create(self, images: List[dict]) -> None:
        if not images:
            return
        objects = [ImageAsset(**img) for img in images]
        self.db.add_all(objects)
        self.db.commit()

    def get_by_document(self, document_id: str) -> List[ImageAsset]:
        return (
            self.db.query(ImageAsset)
            .filter(ImageAsset.document_id == document_id)
            .order_by(ImageAsset.page_number)
            .all()
        )

    def get(self, image_id: str) -> Optional[ImageAsset]:
        return self.db.query(ImageAsset).filter(ImageAsset.id == image_id).first()

    def delete_by_document(self, document_id: str) -> None:
        self.db.query(ImageAsset).filter(ImageAsset.document_id == document_id).delete()
        self.db.commit()


class ExtractedTableRepository:
    def __init__(self, db: Session):
        self.db = db

    def bulk_create(self, tables: List[dict]) -> None:
        if not tables:
            return
        objects = [ExtractedTable(**t) for t in tables]
        self.db.add_all(objects)
        self.db.commit()

    def get_by_document(self, document_id: str) -> List[ExtractedTable]:
        return (
            self.db.query(ExtractedTable)
            .filter(ExtractedTable.document_id == document_id)
            .order_by(ExtractedTable.page_number)
            .all()
        )

    def delete_by_document(self, document_id: str) -> None:
        self.db.query(ExtractedTable).filter(ExtractedTable.document_id == document_id).delete()
        self.db.commit()
