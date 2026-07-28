import io
import time


def _make_fake_pdf_bytes(text: str = "This document discusses cloud computing and container orchestration.") -> bytes:
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_upload_requires_authentication(client):
    files = {"files": ("research.pdf", io.BytesIO(_make_fake_pdf_bytes()), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 401


def test_upload_rejects_non_pdf_file(auth_client):
    files = {"files": ("notes.txt", io.BytesIO(b"just some text"), "text/plain")}
    response = auth_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_upload_and_list_document(auth_client):
    pdf_bytes = _make_fake_pdf_bytes()
    files = {"files": ("research.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    assert upload_response.status_code == 200
    body = upload_response.json()
    assert body["success"] is True
    assert len(body["results"]) == 1
    document_id = body["results"][0]["document_id"]

    # Background processing runs via TestClient's task runner synchronously after the response;
    # give it a brief moment in case of any scheduling delay.
    time.sleep(0.5)

    list_response = auth_client.get("/api/v1/documents")
    assert list_response.status_code == 200
    list_body = list_response.json()
    assert list_body["count"] >= 1
    assert any(d["document_id"] == document_id for d in list_body["documents"])


def test_get_nonexistent_document_returns_404(auth_client):
    response = auth_client.get("/api/v1/documents/does-not-exist")
    assert response.status_code == 404
    assert response.json()["success"] is False


def test_delete_nonexistent_document_returns_404(auth_client):
    response = auth_client.delete("/api/v1/documents/does-not-exist")
    assert response.status_code == 404


def test_user_cannot_view_another_users_document(auth_client, second_auth_client):
    pdf_bytes = _make_fake_pdf_bytes()
    files = {"files": ("owned_by_user_a.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]

    # User B must not be able to see User A's document.
    response = second_auth_client.get(f"/api/v1/documents/{document_id}")
    assert response.status_code == 404


def test_user_cannot_delete_another_users_document(auth_client, second_auth_client):
    pdf_bytes = _make_fake_pdf_bytes()
    files = {"files": ("owned_by_user_a2.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]

    response = second_auth_client.delete(f"/api/v1/documents/{document_id}")
    assert response.status_code == 404

    # And it must still exist for its actual owner.
    still_there = auth_client.get(f"/api/v1/documents/{document_id}")
    assert still_there.status_code == 200


def test_user_document_list_does_not_include_other_users_documents(auth_client, second_auth_client):
    pdf_bytes = _make_fake_pdf_bytes()
    files = {"files": ("user_a_only.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]

    list_response = second_auth_client.get("/api/v1/documents")
    assert list_response.status_code == 200
    ids = [d["document_id"] for d in list_response.json()["documents"]]
    assert document_id not in ids
