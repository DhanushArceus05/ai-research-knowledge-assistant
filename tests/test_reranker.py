from app.vector_store.reranker import Reranker


def test_reranker_changes_order_when_model_available(monkeypatch):
    """Mocks the CrossEncoder so the test never downloads a real model, then
    verifies that applying scores actually re-sorts the candidate list."""
    reranker = Reranker()

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            # Deliberately score the *second* candidate highest, to prove reordering occurs.
            return [0.1, 0.9, 0.3]

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder", FakeCrossEncoder
    )

    candidates = [
        {"text": "first candidate", "document_id": "d1", "page_number": 1},
        {"text": "second candidate", "document_id": "d1", "page_number": 2},
        {"text": "third candidate", "document_id": "d1", "page_number": 3},
    ]

    reranked, applied, reason = reranker.rerank("some query", candidates, top_k=3)

    assert applied is True
    assert reranked[0]["text"] == "second candidate"
    assert reranked[0]["rerank_score"] == 0.9
    assert "rerank_score" in reranked[1]


def test_reranker_respects_top_k():
    reranker = Reranker()
    reranker.enabled = False  # force fallback path (no model load attempted)

    candidates = [
        {"text": "a", "document_id": "d1", "page_number": 1},
        {"text": "b", "document_id": "d1", "page_number": 2},
        {"text": "c", "document_id": "d1", "page_number": 3},
    ]
    reranked, applied, reason = reranker.rerank("query", candidates, top_k=2)

    assert applied is False
    assert len(reranked) == 2


def test_reranker_falls_back_gracefully_when_disabled():
    reranker = Reranker()
    reranker.enabled = False

    candidates = [{"text": "x", "document_id": "d1", "page_number": 1}]
    reranked, applied, reason = reranker.rerank("query", candidates, top_k=5)

    assert applied is False
    assert "disabled" in reason.lower()
    assert reranked == candidates


def test_reranker_falls_back_gracefully_when_model_load_fails(monkeypatch):
    """Simulates a model download/initialization failure (e.g. no internet access
    to Hugging Face) and verifies the reranker reports the fallback instead of crashing."""
    reranker = Reranker()

    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated network failure downloading model")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", raise_error)

    candidates = [{"text": "x", "document_id": "d1", "page_number": 1}]
    reranked, applied, reason = reranker.rerank("query", candidates, top_k=5)

    assert applied is False
    assert "falling back" in reason.lower()
    assert reranked == candidates


def test_reranker_handles_empty_candidate_list():
    reranker = Reranker()
    reranked, applied, reason = reranker.rerank("query", [], top_k=5)
    assert reranked == []
    assert applied is False
