"""
Shared pytest fixtures. Uses an isolated temp environment (SQLite file, upload dir,
vector db dir) per test session and mocks heavy external dependencies (Gemini,
sentence-transformers, ChromaDB, TensorFlow) so the test suite runs fast and
never makes real network/API calls.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

import pytest

# Point the app at an isolated temp workspace before any app modules are imported.
_TEST_DIR = Path(tempfile.mkdtemp(prefix="ai_assistant_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DIR}/test_app.db"
os.environ["UPLOAD_DIR"] = str(_TEST_DIR / "raw_documents")
os.environ["VECTOR_DB_DIR"] = str(_TEST_DIR / "vector_db")
os.environ["MODEL_PATH"] = str(_TEST_DIR / "models" / "document_classifier.keras")
os.environ["VECTORIZER_PATH"] = str(_TEST_DIR / "models" / "text_vectorizer.pkl")
os.environ["LABEL_ENCODER_PATH"] = str(_TEST_DIR / "models" / "label_encoder.pkl")
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["DEBUG"] = "false"

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session", autouse=True)
def _cleanup_temp_dir():
    yield
    shutil.rmtree(_TEST_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _mock_embedding_service(monkeypatch):
    """Replaces the sentence-transformers model with a deterministic fake so no
    real model download / inference happens during tests."""
    from app.vector_store import embeddings as embeddings_module

    def fake_embed_texts(self, texts):
        # Deterministic pseudo-embedding based on text length/hash, fixed dimensionality.
        return [[float((hash(t) % 1000)) / 1000.0] * 8 for t in texts]

    monkeypatch.setattr(embeddings_module.EmbeddingService, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        embeddings_module.EmbeddingService,
        "embed_query",
        lambda self, q: fake_embed_texts(self, [q])[0],
    )
    embeddings_module.get_embedding_service.cache_clear()
    yield
    embeddings_module.get_embedding_service.cache_clear()


@pytest.fixture(autouse=True)
def _mock_gemini_client(monkeypatch):
    """Replaces the Gemini client's generate()/generate_stream() calls so tests
    never hit the real API."""
    from app.rag import gemini_client as gemini_module

    def fake_generate(self, prompt, temperature=0.0):
        return "This is a mocked response used for testing purposes."

    def fake_generate_stream(self, prompt, temperature=0.0):
        for word in ["This ", "is ", "a ", "mocked ", "streamed ", "response."]:
            yield word

    monkeypatch.setattr(gemini_module.GeminiClient, "generate", fake_generate)
    monkeypatch.setattr(gemini_module.GeminiClient, "generate_stream", fake_generate_stream)
    gemini_module.get_gemini_client.cache_clear()
    yield
    gemini_module.get_gemini_client.cache_clear()


@pytest.fixture(autouse=True)
def _clear_cache_manager():
    """Clears the process-wide cache singleton before each test so cached
    summaries/searches/analytics from one test never leak into another."""
    from app.core.cache import get_cache_manager
    cache = get_cache_manager()
    cache._memory_cache.clear()
    yield
    cache._memory_cache.clear()


@pytest.fixture()
def client():
    """FastAPI TestClient with a freshly initialized database for each test."""
    from fastapi.testclient import TestClient
    from app.database.base import init_db
    from app.main import app

    init_db()
    with TestClient(app) as test_client:
        yield test_client


_user_counter = {"n": 0}


def _register_new_user(client, password: str = "TestPass123!"):
    """Registers a brand-new unique user against the given TestClient and returns (user_id, token, email)."""
    _user_counter["n"] += 1
    email = f"testuser{_user_counter['n']}@example.com"
    response = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    body = response.json()
    return body["user"]["user_id"], body["access_token"], email


@pytest.fixture()
def auth_client(client):
    """A TestClient pre-configured with a valid Bearer token for a freshly registered user."""
    user_id, token, email = _register_new_user(client)
    client.headers.update({"Authorization": f"Bearer {token}"})
    client.current_user_id = user_id
    client.current_user_email = email
    return client


@pytest.fixture()
def second_auth_client(client):
    """A second TestClient (same app/db) authenticated as a different user, for isolation tests."""
    from fastapi.testclient import TestClient
    from app.main import app

    second_client = TestClient(app)
    user_id, token, email = _register_new_user(second_client)
    second_client.headers.update({"Authorization": f"Bearer {token}"})
    second_client.current_user_id = user_id
    second_client.current_user_email = email
    return second_client
