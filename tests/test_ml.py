from app.ml.predictor import DocumentClassifier, UNCLASSIFIED


def test_predictor_falls_back_to_unclassified_when_model_files_missing():
    classifier = DocumentClassifier()
    classifier.model_path = "/nonexistent/path/model.keras"
    classifier.label_encoder_path = "/nonexistent/path/label_encoder.pkl"

    category, confidence = classifier.predict("Some document text about cloud computing.")

    assert category == UNCLASSIFIED
    assert confidence == 0.0


def test_predictor_handles_empty_text_gracefully():
    classifier = DocumentClassifier()
    category, confidence = classifier.predict("")

    assert category == UNCLASSIFIED
    assert confidence == 0.0


def test_classify_endpoint_requires_authentication(client):
    response = client.post("/api/v1/analysis/classify", json={"document_id": "does-not-exist"})
    assert response.status_code == 401


def test_classify_endpoint_returns_404_for_missing_document(auth_client):
    response = auth_client.post("/api/v1/analysis/classify", json={"document_id": "does-not-exist"})
    assert response.status_code == 404


def test_classify_endpoint_rejects_empty_document_id(auth_client):
    response = auth_client.post("/api/v1/analysis/classify", json={"document_id": ""})
    assert response.status_code == 422
