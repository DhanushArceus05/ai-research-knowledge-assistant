"""
Retrieval-Augmented Generation question answering service.

Supports three retrieval modes, always scoped to the requesting user's own
documents:
  - "semantic" (default): dense vector similarity via ChromaDB.
  - "keyword": exact/substring SQLite search.
  - "hybrid": BM25 + vector fusion (Reciprocal Rank Fusion), optionally
    followed by cross-encoder reranking.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.rag.prompts import QA_PROMPT_TEMPLATE, FALLBACK_ANSWER
from app.rag.gemini_client import get_gemini_client
from app.vector_store.chroma_manager import get_chroma_manager
from app.vector_store.reranker import get_reranker
from app.services.search_service import SearchService
from app.services.hybrid_search_service import HybridSearchService
from app.core.logging import get_logger

logger = get_logger(__name__)

RETRIEVAL_MODES = ("semantic", "keyword", "hybrid")


class QuestionAnsweringService:
    def __init__(self, db: Optional[Session] = None):
        self.settings = get_settings()
        self.db = db
        self.chroma = get_chroma_manager()
        self.gemini = get_gemini_client()

    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return "(no prior conversation)"
        lines = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def _deduplicate_citations(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        citations = []
        for item in retrieved:
            key = (item["file_name"], item["page_number"])
            if key not in seen:
                seen.add(key)
                citations.append({"document": item["file_name"], "page": item["page_number"]})
        return citations

    def _retrieve(
        self,
        question: str,
        user_id: Optional[str],
        document_ids: Optional[List[str]],
        top_k: int,
        retrieval_mode: str,
        apply_reranking: bool,
    ) -> List[Dict[str, Any]]:
        if retrieval_mode == "keyword":
            if self.db is None:
                raise ValueError("A database session is required for keyword retrieval mode.")
            search_service = SearchService(self.db)
            raw = search_service.keyword_search(question, user_id=user_id, document_ids=document_ids)
            # SQLite substring search has no natural relevance ranking; truncate to top_k.
            return raw[:top_k]

        if retrieval_mode == "hybrid":
            if self.db is None:
                raise ValueError("A database session is required for hybrid retrieval mode.")
            hybrid_service = HybridSearchService(self.db)
            candidate_k = top_k * 2 if apply_reranking else top_k
            candidates = hybrid_service.search(question, user_id=user_id, top_k=candidate_k, document_ids=document_ids)
            if apply_reranking and candidates:
                reranker = get_reranker()
                candidates, _, _ = reranker.rerank(question, candidates, top_k)
            else:
                candidates = candidates[:top_k]
            # Normalize field name for downstream code (hybrid uses "fused_score", not "similarity").
            for c in candidates:
                c.setdefault("similarity", c.get("fused_score"))
            return candidates

        # Default: semantic
        return self.chroma.semantic_search(question, top_k=top_k, document_ids=document_ids, user_id=user_id)

    def answer_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
        retrieval_mode: str = "semantic",
        apply_reranking: bool = False,
    ):
        """
        Generator yielding dicts of the form {"event": "token"|"metadata"|"error"|"done", "data": {...}}.
        Callers are responsible for formatting these as Server-Sent Events. The full
        accumulated answer text is included in the final "metadata" event's data
        under the "answer" key, so callers can persist the completed message
        (never persist a partial answer if the stream errors out mid-generation).
        """
        k = top_k or self.settings.TOP_K_RESULTS
        mode = retrieval_mode if retrieval_mode in RETRIEVAL_MODES else "semantic"

        try:
            retrieved = self._retrieve(question, user_id, document_ids, k, mode, apply_reranking)
        except Exception as exc:
            yield {"event": "error", "data": {"success": False, "message": f"Retrieval failed: {exc}"}}
            return

        if not retrieved:
            yield {"event": "token", "data": {"text": FALLBACK_ANSWER}}
            yield {
                "event": "metadata",
                "data": {
                    "answer": FALLBACK_ANSWER,
                    "citations": [],
                    "confidence": None,
                    "retrieval_mode": mode,
                    "source_documents": [],
                },
            }
            yield {"event": "done", "data": {"success": True}}
            return

        context_str = "\n".join(
            f"\n--- Source: {r['file_name']} (Page {r['page_number']}) ---\n{r['text']}\n"
            for r in retrieved
        )
        prompt = QA_PROMPT_TEMPLATE.format(
            fallback=FALLBACK_ANSWER,
            history=self._format_history(history or []),
            context=context_str,
            question=question,
        )

        accumulated = []
        try:
            for text_chunk in self.gemini.generate_stream(prompt):
                accumulated.append(text_chunk)
                yield {"event": "token", "data": {"text": text_chunk}}
        except Exception as exc:
            # Do not emit a metadata/done event for a partial answer; the caller
            # must not persist an incomplete assistant message.
            yield {"event": "error", "data": {"success": False, "message": str(exc)}}
            return

        full_answer = "".join(accumulated).strip()
        similarities = [r["similarity"] for r in retrieved if r.get("similarity") is not None]
        confidence = round(sum(similarities) / len(similarities), 4) if similarities else None

        yield {
            "event": "metadata",
            "data": {
                "answer": full_answer,
                "citations": self._deduplicate_citations(retrieved),
                "confidence": confidence,
                "retrieval_mode": mode,
                "source_documents": sorted({r["file_name"] for r in retrieved}),
            },
        }
        yield {"event": "done", "data": {"success": True}}

    def answer(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        user_id: Optional[str] = None,
        retrieval_mode: str = "semantic",
        apply_reranking: bool = False,
    ) -> Dict[str, Any]:
        k = top_k or self.settings.TOP_K_RESULTS
        mode = retrieval_mode if retrieval_mode in RETRIEVAL_MODES else "semantic"

        retrieved = self._retrieve(question, user_id, document_ids, k, mode, apply_reranking)

        if not retrieved:
            return {
                "answer": FALLBACK_ANSWER,
                "citations": [],
                "confidence": None,
                "retrieved_context": [],
                "retrieval_mode": mode,
            }

        context_str = "\n".join(
            f"\n--- Source: {r['file_name']} (Page {r['page_number']}) ---\n{r['text']}\n"
            for r in retrieved
        )

        prompt = QA_PROMPT_TEMPLATE.format(
            fallback=FALLBACK_ANSWER,
            history=self._format_history(history or []),
            context=context_str,
            question=question,
        )

        answer_text = self.gemini.generate(prompt)

        similarities = [r["similarity"] for r in retrieved if r.get("similarity") is not None]
        confidence = round(sum(similarities) / len(similarities), 4) if similarities else None

        return {
            "answer": answer_text,
            "citations": self._deduplicate_citations(retrieved),
            "confidence": confidence,
            "retrieval_mode": mode,
            "retrieved_context": [
                {
                    "document_id": r["document_id"],
                    "file_name": r["file_name"],
                    "page_number": r["page_number"],
                    "text": r["text"],
                    "similarity": r.get("similarity"),
                }
                for r in retrieved
            ],
        }
