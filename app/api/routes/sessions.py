from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.services.conversation_service import ConversationService
from app.schemas.chat import SessionHistoryResponse, SessionMessage, SessionDeleteResponse

router = APIRouter(prefix="/sessions", tags=["Conversation Sessions"])


@router.get("/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retrieves the full conversation history for a session owned by the current user."""
    service = ConversationService(db)
    messages = service.get_session_history(session_id, current_user)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[SessionMessage(**m) for m in messages],
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
def clear_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Clears/deletes a conversation session owned by the current user."""
    service = ConversationService(db)
    service.clear_session(session_id, current_user)
    return SessionDeleteResponse(message=f"Session '{session_id}' cleared successfully.")
