from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.database.models import User
from app.services.agent_service import ResearchAgent
from app.services.conversation_service import ConversationService
from app.schemas.agent import AgentRunRequest, AgentRunResponse

router = APIRouter(prefix="/agent", tags=["Research Agent"])


@router.post("/run", response_model=AgentRunResponse)
def run_agent(payload: AgentRunRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Runs the deterministic research agent: routes the instruction to an allowlisted
    tool (list_documents, document_metadata, document_search, hybrid_search,
    answer_from_documents, summarize_document, compare_documents, analytics),
    executes it with the current user's ownership scope enforced, and returns a
    safe high-level execution trace alongside the final grounded result.
    """
    conversation_service = ConversationService(db)
    session = conversation_service.get_or_create_session(payload.session_id, current_user)

    agent = ResearchAgent(db, current_user)
    result = agent.run(
        instruction=payload.instruction,
        document_ids=payload.document_ids,
        session_id=session.id,
        top_k=payload.top_k,
    )

    return AgentRunResponse(
        success=result["success"],
        selected_tool=result.get("selected_tool"),
        trace=result["trace"],
        answer=result.get("answer"),
        citations=result.get("citations", []),
        session_id=session.id,
        error=result.get("error"),
    )
