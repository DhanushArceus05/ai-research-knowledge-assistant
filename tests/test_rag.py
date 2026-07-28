import pytest

from app.rag.qa_service import QuestionAnsweringService
from app.rag.prompts import FALLBACK_ANSWER


def test_qa_service_returns_fallback_when_no_context_retrieved(monkeypatch):
    service = QuestionAnsweringService()

    # Force the vector store to return no results, simulating an empty/irrelevant knowledge base.
    monkeypatch.setattr(service.chroma, "semantic_search", lambda *args, **kwargs: [])

    result = service.answer(question="What is the meaning of life?")

    assert result["answer"] == FALLBACK_ANSWER
    assert result["citations"] == []
    assert result["retrieved_context"] == []


def test_qa_service_deduplicates_citations_from_same_source(monkeypatch):
    service = QuestionAnsweringService()

    fake_chunks = [
        {"text": "chunk A", "document_id": "d1", "file_name": "paper.pdf", "page_number": 2,
         "chunk_index": 0, "similarity": 0.9},
        {"text": "chunk B", "document_id": "d1", "file_name": "paper.pdf", "page_number": 2,
         "chunk_index": 1, "similarity": 0.85},
    ]
    monkeypatch.setattr(service.chroma, "semantic_search", lambda *args, **kwargs: fake_chunks)

    result = service.answer(question="Summarize the key point.")

    # Both chunks share the same (file_name, page_number) so only one citation should remain.
    assert len(result["citations"]) == 1
    assert result["citations"][0]["document"] == "paper.pdf"
    assert result["citations"][0]["page"] == 2


def test_qa_service_returns_average_confidence_when_context_retrieved(monkeypatch):
    service = QuestionAnsweringService()

    fake_chunks = [
        {"text": "chunk A", "document_id": "d1", "file_name": "paper.pdf", "page_number": 1,
         "chunk_index": 0, "similarity": 0.8},
        {"text": "chunk B", "document_id": "d1", "file_name": "paper.pdf", "page_number": 2,
         "chunk_index": 1, "similarity": 0.6},
    ]
    monkeypatch.setattr(service.chroma, "semantic_search", lambda *args, **kwargs: fake_chunks)

    result = service.answer(question="What is discussed?")

    assert result["confidence"] == pytest.approx(0.7, abs=1e-6)


def test_qa_service_confidence_is_none_on_fallback(monkeypatch):
    service = QuestionAnsweringService()
    monkeypatch.setattr(service.chroma, "semantic_search", lambda *args, **kwargs: [])

    result = service.answer(question="Anything?")
    assert result["confidence"] is None


def test_chat_endpoint_requires_authentication(client):
    response = client.post("/api/v1/chat/ask", json={"question": "What does this document say?"})
    assert response.status_code == 401


def test_chat_endpoint_creates_and_returns_session_id(auth_client):
    response = auth_client.post("/api/v1/chat/ask", json={"question": "What does this document say?"})
    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body and body["session_id"]
    assert "answer" in body


def test_chat_endpoint_rejects_empty_question(auth_client):
    response = auth_client.post("/api/v1/chat/ask", json={"question": "   "})
    assert response.status_code == 422


def test_chat_endpoint_rejects_invalid_retrieval_mode(auth_client):
    response = auth_client.post("/api/v1/chat/ask", json={"question": "test", "retrieval_mode": "not_a_real_mode"})
    assert response.status_code == 422


def test_stream_endpoint_requires_authentication(client):
    response = client.post("/api/v1/chat/stream", json={"question": "Anything?"})
    assert response.status_code == 401


def test_stream_endpoint_emits_token_metadata_and_done_events(auth_client):
    with auth_client.stream("POST", "/api/v1/chat/stream", json={"question": "What does this document say?"}) as response:
        assert response.status_code == 200
        raw = "".join(response.iter_text())

    assert "event: token" in raw
    assert "event: metadata" in raw
    assert "event: done" in raw
    assert '"session_id"' in raw


def test_stream_endpoint_emits_fallback_token_when_no_context(auth_client, monkeypatch):
    from app.rag import qa_service as qa_module
    # Force an empty retrieval so the fallback sentence path is exercised without a real upload.
    monkeypatch.setattr(qa_module.QuestionAnsweringService, "_retrieve", lambda self, *a, **k: [])

    with auth_client.stream("POST", "/api/v1/chat/stream", json={"question": "Anything?"}) as response:
        raw = "".join(response.iter_text())

    assert "I cannot determine the answer from the provided documents." in raw
    assert "event: done" in raw
