"""
Deterministic research-agent orchestration layer.

Routes an instruction to one of a small set of allowlisted tools using
keyword-based intent matching (no model call needed for routing itself).
Tool calls are capped and time-bounded to avoid runaway loops, and
answer_from_documents always goes through the regular QuestionAnsweringService
so citations and the fallback behavior stay consistent with /chat/ask.
Ownership checks are enforced per tool, same as the rest of the API.
"""
import time
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.models import User
from app.database.repositories import DocumentRepository, ChunkRepository
from app.services.document_service import DocumentService
from app.services.search_service import SearchService
from app.services.hybrid_search_service import HybridSearchService
from app.services.analytics_service import AnalyticsService
from app.rag.qa_service import QuestionAnsweringService
from app.rag.summarizer import SummarizationService
from app.rag.comparator import ComparisonService

logger = get_logger(__name__)

ALLOWED_TOOLS = (
    "list_documents",
    "document_metadata",
    "document_search",
    "hybrid_search",
    "answer_from_documents",
    "summarize_document",
    "compare_documents",
    "analytics",
)

AGENT_TIMEOUT_SECONDS = 25


class AgentToolError(Exception):
    pass


class ResearchAgent:
    def __init__(self, db: Session, user: User):
        self.db = db
        self.user = user
        self.settings = get_settings()
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.doc_service = DocumentService(db)

    # ---- Deterministic intent routing -------------------------------------------------

    def _classify_intent(self, instruction: str, document_ids: Optional[List[str]]) -> str:
        text = instruction.lower()

        if any(w in text for w in ("compare", "difference between", "versus", " vs ")) and document_ids and len(document_ids) >= 2:
            return "compare_documents"
        if any(w in text for w in ("summarize", "summary", "tl;dr", "key takeaways")):
            return "summarize_document"
        if any(w in text for w in ("how many documents", "list my documents", "what documents", "which documents")):
            return "list_documents"
        if any(w in text for w in ("analytics", "usage stats", "how many questions", "statistics")):
            return "analytics"
        if any(w in text for w in ("metadata", "status of", "processing status", "how many pages")):
            return "document_metadata"
        if any(w in text for w in ("hybrid", "bm25", "keyword and semantic")):
            return "hybrid_search"
        return "answer_from_documents"

    # ---- Individual tools (schema-validated, allowlisted) -----------------------------

    def _tool_list_documents(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        documents = self.doc_service.list_documents(self.user.id)
        return {
            "count": len(documents),
            "documents": [
                {"document_id": d.id, "file_name": d.original_file_name, "status": d.processing_status.value}
                for d in documents
            ],
        }

    def _tool_document_metadata(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        if not document_ids:
            raise AgentToolError("document_metadata requires at least one document_id.")
        result = [self.doc_service.get_owned_document(doc_id, self.user).to_dict() for doc_id in document_ids]
        return {"documents": result}

    def _tool_document_search(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        service = SearchService(self.db)
        results = service.semantic_search(instruction, user_id=self.user.id, document_ids=document_ids, top_k=top_k)
        return {"result_count": len(results), "results": results}

    def _tool_hybrid_search(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        service = HybridSearchService(self.db)
        results = service.search(instruction, user_id=self.user.id, top_k=top_k, document_ids=document_ids)
        return {"result_count": len(results), "results": results}

    def _tool_answer_from_documents(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        qa_service = QuestionAnsweringService(self.db)
        result = qa_service.answer(
            question=instruction, document_ids=document_ids, top_k=top_k,
            user_id=self.user.id, retrieval_mode="hybrid",
        )
        return result

    def _tool_summarize_document(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        if not document_ids:
            raise AgentToolError("summarize_document requires exactly one document_id.")
        document = self.doc_service.get_owned_document(document_ids[0], self.user)
        chunks = self.chunk_repo.get_by_document(document.id)
        if not chunks:
            raise AgentToolError("This document has no processed chunks yet.")
        service = SummarizationService()
        result = service.summarize_document(document.original_file_name, [c.text for c in chunks])
        return {"document_id": document.id, "file_name": document.original_file_name, "summary": result["summary"]}

    def _tool_compare_documents(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        if not document_ids or len(document_ids) < 2:
            raise AgentToolError("compare_documents requires at least two document_ids.")
        documents = self.doc_repo.get_many(document_ids, user_id=self.user.id)
        found_ids = {d.id for d in documents}
        missing = [d for d in document_ids if d not in found_ids]
        if missing:
            raise AgentToolError(f"Document(s) not found or not owned by you: {', '.join(missing)}")

        comparison_input = []
        for document in documents:
            chunks = self.chunk_repo.get_by_document(document.id)
            comparison_input.append({
                "file_name": document.original_file_name,
                "chunks": [{"page_number": c.page_number, "text": c.text} for c in chunks],
            })
        service = ComparisonService()
        comparison_text = service.compare_documents(comparison_input)
        return {"document_ids": [d.id for d in documents], "comparison": comparison_text}

    def _tool_analytics(self, instruction, document_ids, session_id, top_k) -> Dict[str, Any]:
        service = AnalyticsService(self.db)
        return service.get_analytics(user_id=self.user.id)

    _TOOL_DISPATCH = {
        "list_documents": _tool_list_documents,
        "document_metadata": _tool_document_metadata,
        "document_search": _tool_document_search,
        "hybrid_search": _tool_hybrid_search,
        "answer_from_documents": _tool_answer_from_documents,
        "summarize_document": _tool_summarize_document,
        "compare_documents": _tool_compare_documents,
        "analytics": _tool_analytics,
    }

    # ---- Orchestration ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        document_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time = time.monotonic()
        max_steps = self.settings.AGENT_MAX_STEPS
        trace: List[Dict[str, Any]] = []
        step = 1
        k = top_k or self.settings.TOP_K_RESULTS

        intent = self._classify_intent(instruction, document_ids)
        if intent not in ALLOWED_TOOLS:
            # Belt and suspenders: _classify_intent only returns allowlisted
            # values today, but this guards against a future routing bug.
            raise AgentToolError(f"Tool '{intent}' is not in the allowlist.")

        trace.append({"step": step, "action": intent, "status": "started"})

        if time.monotonic() - start_time > AGENT_TIMEOUT_SECONDS:
            trace[-1]["status"] = "timeout"
            return {"success": False, "trace": trace, "error": "Agent execution timed out.", "citations": [], "answer": None, "selected_tool": intent}

        tool_fn = self._TOOL_DISPATCH[intent]
        try:
            result = tool_fn(self, instruction, document_ids, session_id, k)
            trace[-1]["status"] = "completed"
        except AgentToolError as exc:
            trace[-1]["status"] = "failed"
            return {"success": False, "trace": trace, "error": str(exc), "citations": [], "answer": None, "selected_tool": intent}
        except Exception as exc:
            logger.exception("Agent tool '%s' raised an unexpected error.", intent)
            trace[-1]["status"] = "failed"
            return {"success": False, "trace": trace, "error": f"Tool execution failed: {exc}", "citations": [], "answer": None, "selected_tool": intent}

        step += 1
        if step > max_steps:
            trace.append({"step": step, "action": "max_steps_reached", "status": "stopped"})
            step -= 1  # do not advance further

        if intent == "answer_from_documents":
            answer = result.get("answer")
            citations = result.get("citations", [])
        elif intent == "summarize_document":
            answer = result.get("summary")
            citations = []
        elif intent == "compare_documents":
            answer = result.get("comparison")
            citations = []
        else:
            answer = None
            citations = []

        trace.append({"step": step, "action": "generate_final_response", "status": "completed"})

        return {
            "success": True,
            "selected_tool": intent,
            "trace": trace,
            "answer": answer,
            "citations": citations,
            "result": result,
        }
