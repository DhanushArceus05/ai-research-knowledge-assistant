"""
Persistent ChromaDB vector store manager: indexing, semantic search, and deletion.
"""
from typing import List, Dict, Any, Optional
from functools import lru_cache

import chromadb

from app.core.config import get_settings
from app.core.logging import get_logger
from app.vector_store.embeddings import get_embedding_service

logger = get_logger(__name__)

COLLECTION_NAME = "document_chunks"


class ChromaManager:
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.VECTOR_DB_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedding_service = get_embedding_service()

    def index_chunks(self, chunks: List[Dict[str, Any]], file_name: str, user_id: Optional[str] = None) -> None:
        """Embeds and stores a batch of chunk dicts (document_id, chunk_index, page_number, text)."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = self.embedding_service.embed_texts(texts)

        ids = [f"{c['document_id']}_c{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "document_id": c["document_id"],
                "file_name": file_name,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "user_id": user_id or "",
            }
            for c in chunks
        ]

        self.collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    def semantic_search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_service.embed_query(query)

        conditions = []
        if document_ids:
            conditions.append({"document_id": {"$in": document_ids}})
        if user_id is not None:
            conditions.append({"user_id": user_id})

        if len(conditions) == 0:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, meta, distance in zip(docs, metas, distances):
            similarity = 1.0 / (1.0 + distance) if distance is not None else None
            output.append({
                "text": text,
                "document_id": meta.get("document_id"),
                "file_name": meta.get("file_name"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index"),
                "similarity": similarity,
            })
        return output

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id is None:
            return self.collection.count()
        try:
            result = self.collection.get(where={"user_id": user_id})
            return len(result.get("ids", []))
        except Exception:
            return 0


@lru_cache
def get_chroma_manager() -> ChromaManager:
    return ChromaManager()
