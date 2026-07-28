def test_compare_requires_authentication(client):
    response = client.post("/api/v1/analysis/compare", json={"document_ids": ["a", "b"]})
    assert response.status_code == 401


def test_compare_requires_at_least_two_documents(auth_client):
    response = auth_client.post("/api/v1/analysis/compare", json={"document_ids": ["only-one-id"]})
    assert response.status_code == 422


def test_compare_requires_document_ids_field(auth_client):
    response = auth_client.post("/api/v1/analysis/compare", json={"document_ids": []})
    assert response.status_code == 422


def test_compare_with_nonexistent_documents_returns_404(auth_client):
    response = auth_client.post(
        "/api/v1/analysis/compare",
        json={"document_ids": ["missing-1", "missing-2"]},
    )
    assert response.status_code == 404


def test_summarize_requires_document_id(auth_client):
    response = auth_client.post("/api/v1/analysis/summarize", json={"document_id": ""})
    assert response.status_code == 422


def test_summarize_nonexistent_document_returns_404(auth_client):
    response = auth_client.post("/api/v1/analysis/summarize", json={"document_id": "does-not-exist"})
    assert response.status_code == 404


def test_user_cannot_summarize_another_users_document(auth_client, second_auth_client):
    import io, time
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Some content about robotics and sensor fusion.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("robotics.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]
    time.sleep(0.5)

    response = second_auth_client.post("/api/v1/analysis/summarize", json={"document_id": document_id})
    assert response.status_code == 404


def test_user_cannot_compare_using_another_users_document(auth_client, second_auth_client):
    import io, time
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Content about natural language processing.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("nlp.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id_a = upload_response.json()["results"][0]["document_id"]
    time.sleep(0.5)

    # User B tries to compare their own (nonexistent) doc id against User A's doc.
    response = second_auth_client.post(
        "/api/v1/analysis/compare", json={"document_ids": [document_id_a, "some-other-id"]}
    )
    assert response.status_code == 404
