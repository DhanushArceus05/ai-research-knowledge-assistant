import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.rag.qa_service import QuestionAnsweringService
from app.services.conversation_service import ConversationService
from app.database.repositories import DocumentRepository, QueryLogRepository
from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Question Answering"])


@router.post("/ask", response_model=ChatResponse)
def ask_question(payload: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Answers a question using Retrieval-Augmented Generation, grounded strictly in
    the current user's own retrieved document context, with citations and
    conversation memory support. Supports selectable retrieval_mode (semantic/
    keyword/hybrid) and optional cross-encoder reranking.
    """
    conversation_service = ConversationService(db)
    session = conversation_service.get_or_create_session(payload.session_id, current_user)
    history = conversation_service.get_recent_history(session.id)

    qa_service = QuestionAnsweringService(db)
    result = qa_service.answer(
        question=payload.question,
        history=history,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        user_id=current_user.id,
        retrieval_mode=payload.retrieval_mode,
        apply_reranking=payload.apply_reranking,
    )

    conversation_service.record_exchange(session.id, payload.question, result["answer"])

    doc_repo = DocumentRepository(db)
    seen_doc_names = {item["file_name"] for item in result["retrieved_context"]}
    for item in result["retrieved_context"]:
        doc_repo.increment_query_count(item["document_id"])

    QueryLogRepository(db).log(payload.question, session.id, payload.document_ids, user_id=current_user.id)

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        source_documents=sorted(seen_doc_names),
        retrieved_context=result["retrieved_context"],
        session_id=session.id,
        confidence=result.get("confidence"),
        retrieval_mode=result.get("retrieval_mode", payload.retrieval_mode),
    )


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream")
def stream_chat(payload: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Streams the RAG answer as Server-Sent Events using Gemini's native streaming
    support. Event types: `token` (incremental text), `metadata` (citations,
    confidence, source documents, session_id once generation completes), `done`,
    and `error` (emitted instead of metadata/done if generation fails partway
    through -- an incomplete answer is never persisted as a completed message).
    """
    settings = get_settings()
    if not settings.STREAMING_ENABLED:
        raise ConfigurationError("Streaming responses are disabled via STREAMING_ENABLED=false.")

    conversation_service = ConversationService(db)
    session = conversation_service.get_or_create_session(payload.session_id, current_user)
    history = conversation_service.get_recent_history(session.id)
    session_id = session.id

    qa_service = QuestionAnsweringService(db)
    doc_repo = DocumentRepository(db)
    query_log_repo = QueryLogRepository(db)

    def event_generator():
        completed_answer = None
        try:
            for event in qa_service.answer_stream(
                question=payload.question,
                history=history,
                document_ids=payload.document_ids,
                top_k=payload.top_k,
                user_id=current_user.id,
                retrieval_mode=payload.retrieval_mode,
                apply_reranking=payload.apply_reranking,
            ):
                if event["event"] == "metadata":
                    completed_answer = event["data"]["answer"]
                    event["data"]["session_id"] = session_id
                yield _sse_event(event["event"], event["data"])
        finally:
            if completed_answer is not None:
                conversation_service.record_exchange(session_id, payload.question, completed_answer)
                query_log_repo.log(payload.question, session_id, payload.document_ids, user_id=current_user.id)
                if payload.document_ids:
                    for doc_id in payload.document_ids:
                        doc_repo.increment_query_count(doc_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
