"""
Conversation memory service backed by SQLite, used to give the RAG QA service
follow-up-question context (e.g. resolving "it" / "its" to a prior document).
All sessions are scoped to the owning user.
"""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import SessionNotFoundError
from app.database.repositories import ConversationRepository
from app.database.models import ConversationSession, User, UserRole


class ConversationService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.repo = ConversationRepository(db)

    def _is_owned_by(self, session: ConversationSession, user: User) -> bool:
        if user.role == UserRole.ADMIN:
            return True
        return session.user_id is not None and session.user_id == user.id

    def get_or_create_session(self, session_id: Optional[str], user: User) -> ConversationSession:
        session = self.repo.get_or_create_session(session_id, user_id=user.id)
        if not self._is_owned_by(session, user):
            # A session_id was supplied that belongs to a different user: never
            # attach someone else's conversation history. Start a brand-new,
            # separately-identified session instead of leaking access.
            raise SessionNotFoundError(session_id or "")
        return session

    def get_recent_history(self, session_id: str) -> List[Dict[str, str]]:
        messages = self.repo.get_recent_messages(session_id, limit=self.settings.MAX_HISTORY_MESSAGES)
        return [{"role": m.role, "content": m.content} for m in messages]

    def record_exchange(self, session_id: str, question: str, answer: str) -> None:
        self.repo.add_message(session_id, "user", question)
        self.repo.add_message(session_id, "assistant", answer)

    def get_session_history(self, session_id: str, user: User) -> List[Dict[str, str]]:
        session = self.repo.get_session(session_id)
        if not session or not self._is_owned_by(session, user):
            raise SessionNotFoundError(session_id)
        return [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in session.messages
        ]

    def clear_session(self, session_id: str, user: User) -> None:
        session = self.repo.get_session(session_id)
        if not session or not self._is_owned_by(session, user):
            raise SessionNotFoundError(session_id)
        self.repo.clear_session(session_id)
