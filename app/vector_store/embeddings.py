"""
Embedding generation wrapper around sentence-transformers.

The model is loaded lazily (on first use) so the API can start quickly and
so import errors / missing model downloads don't crash application startup.
"""
from typing import List
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
