"""
BM25 sparse-retrieval index, built on demand (per user/document scope) from
chunks stored in SQLite. Uses `rank-bm25`, a small, pure-Python, CPU-only
library with no model downloads required.

The index is intentionally rebuilt per request rather than kept as a single
global in-memory structure: this keeps the implementation simple and
correctness-first (always reflects the latest chunks for the given
user/document scope) at a modest CPU cost that is fine for the small/medium
document collections this project targets. For larger deployments this would
be the natural place to introduce an incremental/cached index per user.
"""
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from app.database.repositories import ChunkRepository
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """A BM25 index built from a specific set of chunks (already filtered by ownership)."""

    def __init__(self, chunks: List[Any]):
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self._bm25:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank + 1,
            })
        return results


def build_bm25_index(
    chunk_repo: ChunkRepository, user_id: Optional[str], document_ids: Optional[List[str]] = None
) -> BM25Index:
    chunks = chunk_repo.get_all_for_user(user_id=user_id, document_ids=document_ids)
    return BM25Index(chunks)
