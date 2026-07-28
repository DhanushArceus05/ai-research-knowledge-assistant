import io
import time


def _upload_pdf(client, text: str, filename: str = "doc.pdf"):
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": (filename, io.BytesIO(pdf_bytes), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    document_id = response.json()["results"][0]["document_id"]
    time.sleep(0.5)
    return document_id


def test_agent_requires_authentication(client):
    response = client.post("/api/v1/agent/run", json={"instruction": "list my documents"})
    assert response.status_code == 401


def test_agent_rejects_empty_instruction(auth_client):
    response = auth_client.post("/api/v1/agent/run", json={"instruction": "   "})
    assert response.status_code == 422


def test_agent_routes_to_list_documents(auth_client):
    _upload_pdf(auth_client, "Some content about cyber security incident response.")
    response = auth_client.post("/api/v1/agent/run", json={"instruction": "list my documents please"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["selected_tool"] == "list_documents"
    assert any(step["action"] == "list_documents" for step in body["trace"])


def test_agent_routes_to_analytics(auth_client):
    response = auth_client.post("/api/v1/agent/run", json={"instruction": "show me my usage analytics"})
    assert response.status_code == 200
    assert response.json()["selected_tool"] == "analytics"


def test_agent_routes_to_answer_from_documents_by_default(auth_client):
    response = auth_client.post("/api/v1/agent/run", json={"instruction": "what does this say about the topic?"})
    assert response.status_code == 200
    body = response.json()
    assert body["selected_tool"] == "answer_from_documents"
    assert body["answer"] is not None


def test_agent_summarize_requires_document_id(auth_client):
    response = auth_client.post("/api/v1/agent/run", json={"instruction": "please summarize this for me"})
    assert response.status_code == 200
    body = response.json()
    # Routed correctly to summarize_document, but fails cleanly since no document_ids were given.
    assert body["selected_tool"] == "summarize_document"
    assert body["success"] is False
    assert "document_id" in body["error"]


def test_agent_rejects_unknown_tool_name_at_dispatch_level():
    """Directly exercises the allowlist guard: a hypothetical routing bug that
    returned a non-allowlisted tool name must be rejected rather than executed."""
    from app.services.agent_service import ResearchAgent, AgentToolError
    from app.database.session import get_db

    db = next(get_db())

    class FakeUser:
        id = "fake"
        role = None

    agent = ResearchAgent(db, FakeUser())
    agent._classify_intent = lambda instruction, document_ids: "delete_everything"

    try:
        raised = False
        try:
            agent.run("do something dangerous")
        except AgentToolError:
            raised = True
        assert raised
    finally:
        db.close()


def test_agent_enforces_document_ownership_for_metadata_tool(auth_client, second_auth_client):
    document_id = _upload_pdf(auth_client, "Content owned by user A about databases.")

    response = second_auth_client.post(
        "/api/v1/agent/run",
        json={"instruction": "what is the metadata status of this document", "document_ids": [document_id]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False  # user B does not own this document
