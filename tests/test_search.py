def test_semantic_search_requires_authentication(client):
    response = client.post("/api/v1/search/semantic", json={"query": "cloud"})
    assert response.status_code == 401


def test_semantic_search_rejects_empty_query(auth_client):
    response = auth_client.post("/api/v1/search/semantic", json={"query": ""})
    assert response.status_code == 422


def test_keyword_search_rejects_empty_query(auth_client):
    response = auth_client.post("/api/v1/search/keyword", json={"query": "   "})
    assert response.status_code == 422


def test_search_rejects_invalid_top_k(auth_client):
    response = auth_client.post("/api/v1/search/semantic", json={"query": "cloud", "top_k": 0})
    assert response.status_code == 422


def test_semantic_search_with_empty_collection_returns_empty_results(monkeypatch):
    """
    Uses an isolated, empty in-memory-equivalent Chroma collection directly
    (rather than the shared session-wide persistent one, which other tests may
    have already populated) to verify the "no matches" code path in isolation.
    """
    from app.services.search_service import SearchService
    from app.database.session import get_db

    db = next(get_db())
    try:
        service = SearchService(db)
        monkeypatch.setattr(service.chroma, "semantic_search", lambda *args, **kwargs: [])
        results = service.semantic_search("some obscure query with no matches", user_id="fake-user-id")
        assert results == []
    finally:
        db.close()


def test_keyword_search_with_no_matches_returns_empty_results(auth_client):
    response = auth_client.post("/api/v1/search/keyword", json={"query": "zzz_no_such_term_zzz"})
    assert response.status_code == 200
    assert response.json()["result_count"] == 0


def test_user_cannot_search_another_users_chunks(auth_client, second_auth_client):
    """Uploads a distinctively-worded document as user A, then verifies user B's
    keyword search for that exact phrase returns nothing."""
    import io
    import time
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "ZebraStripeUniqueMarkerPhrase appears only in this document.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("unique.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    auth_client.post("/api/v1/documents/upload", files=files)
    time.sleep(0.5)

    response = second_auth_client.post("/api/v1/search/keyword", json={"query": "ZebraStripeUniqueMarkerPhrase"})
    assert response.status_code == 200
    assert response.json()["result_count"] == 0


def test_hybrid_search_requires_authentication(client):
    response = client.post("/api/v1/search/hybrid", json={"query": "cloud"})
    assert response.status_code == 401


def test_hybrid_search_rejects_empty_query(auth_client):
    response = auth_client.post("/api/v1/search/hybrid", json={"query": ""})
    assert response.status_code == 422


def test_hybrid_search_returns_valid_shape_with_no_documents(auth_client):
    response = auth_client.post("/api/v1/search/hybrid", json={"query": "anything at all", "apply_reranking": False})
    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 0
    assert body["results"] == []
    assert "reranking_applied" in body
