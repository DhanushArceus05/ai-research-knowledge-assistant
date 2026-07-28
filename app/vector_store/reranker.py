"""
Second-stage cross-encoder reranker.

The reranker model is lazily loaded (never at import time) and cached as a
singleton. If the model cannot be downloaded or initialized (no network
access, or the download fails) or reranking is disabled via config,
`rerank()` transparently falls back to the input ordering (the fused
retrieval ranking) and reports that fallback via `applied=False` plus a
human-readable reason, rather than crashing the request.
"""
from functools import lru_cache
from typing import List, Dict, Any, Tuple

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker:
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.RERANKING_ENABLED
        self.model_name = settings.RERANKER_MODEL_NAME
        self._model = None
        self._load_attempted = False
        self._load_failed_reason = None

    def _load(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        if not self.enabled:
            self._load_failed_reason = "Reranking is disabled via RERANKING_ENABLED=false."
            return False

        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder reranker model '%s'...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:
            self._load_failed_reason = f"Reranker model failed to load ({exc}); falling back to fused ranking."
            logger.warning(self._load_failed_reason)
            self._model = None
            return False

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Returns (reranked_candidates, applied, reason). Each candidate dict gets a
        `rerank_score` key added when reranking is applied. When not applied, the
        original candidate order/scores are returned unchanged.
        """
        if not candidates:
            return candidates, False, "No candidates to rerank."

        if not self._load():
            return candidates[:top_k], False, self._load_failed_reason or "Reranker unavailable."

        try:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self._model.predict(pairs)
            for c, score in zip(candidates, scores):
                c["rerank_score"] = float(score)
            reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            return reranked[:top_k], True, "Cross-encoder reranking applied."
        except Exception as exc:
            reason = f"Reranking failed at inference time ({exc}); falling back to fused ranking."
            logger.warning(reason)
            return candidates[:top_k], False, reason


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
