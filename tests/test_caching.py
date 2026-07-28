import io
import time

from app.core.cache import CacheManager, make_cache_key


def test_make_cache_key_is_deterministic_and_user_scoped():
    key1 = make_cache_key("summary", "user-a", document_id="doc-1")
    key2 = make_cache_key("summary", "user-a", document_id="doc-1")
    key3 = make_cache_key("summary", "user-b", document_id="doc-1")

    assert key1 == key2  # same inputs -> same key (cache hit)
    assert key1 != key3  # different user -> different key (no cross-user leakage)
    assert key1.startswith("summary:user-a:")
    assert key3.startswith("summary:user-b:")


def test_cache_manager_set_and_get_hit():
    cache = CacheManager()
    key = make_cache_key("test_ns", "user-x", foo="bar")

    assert cache.get(key) is None  # miss before set
    cache.set(key, {"value": 42})
    assert cache.get(key) == {"value": 42}  # hit after set


def test_cache_manager_miss_for_unknown_key():
    cache = CacheManager()
    assert cache.get("this-key-was-never-set") is None


def test_cache_manager_delete_prefix_invalidates_only_matching_keys():
    cache = CacheManager()
    key_a = make_cache_key("summary", "user-1", document_id="doc-a")
    key_b = make_cache_key("summary", "user-2", document_id="doc-b")
    cache.set(key_a, "summary for user 1")
    cache.set(key_b, "summary for user 2")

    removed = cache.delete_prefix("summary:user-1:")

    assert removed == 1
    assert cache.get(key_a) is None
    assert cache.get(key_b) == "summary for user 2"  # untouched


def test_cache_backend_falls_back_to_memory_when_redis_unreachable(monkeypatch):
    """Configures CACHE_BACKEND=redis with an unreachable URL and verifies the
    manager falls back to the in-process memory cache instead of crashing."""
    import os
    os.environ["CACHE_BACKEND"] = "redis"
    os.environ["REDIS_URL"] = "redis://nonexistent-host-for-test:6390/0"
    try:
        cache = CacheManager()
        assert cache.backend_mode == "memory"
        # Confirm it still works end-to-end via the memory fallback.
        key = make_cache_key("fallback_test", "user-z")
        cache.set(key, "still works")
        assert cache.get(key) == "still works"
    finally:
        os.environ["CACHE_BACKEND"] = "memory"
        os.environ.pop("REDIS_URL", None)


def test_cache_diagnostics_reports_backend_mode():
    cache = CacheManager()
    diagnostics = cache.diagnostics()
    assert diagnostics["backend_mode"] in ("memory", "redis")
    assert "ttl_seconds" in diagnostics
    assert "memory_cache_size" in diagnostics


def test_summarize_response_is_served_from_cache_on_second_call(auth_client, monkeypatch):
    """Uploads a document, summarizes it twice, and verifies the summarizer's
    underlying Gemini call only happens once (second call is served from cache)."""
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Content about serverless cloud computing architectures.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("cache_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]
    time.sleep(0.5)

    call_count = {"n": 0}
    from app.rag import gemini_client as gemini_module

    def counting_generate(self, prompt, temperature=0.0):
        call_count["n"] += 1
        return "mocked summary text"

    monkeypatch.setattr(gemini_module.GeminiClient, "generate", counting_generate)

    first = auth_client.post("/api/v1/analysis/summarize", json={"document_id": document_id})
    assert first.status_code == 200
    calls_after_first = call_count["n"]
    assert calls_after_first >= 1

    second = auth_client.post("/api/v1/analysis/summarize", json={"document_id": document_id})
    assert second.status_code == 200
    assert call_count["n"] == calls_after_first  # no new Gemini call: served from cache
    assert second.json()["summary"] == first.json()["summary"]


def test_summarize_cache_is_invalidated_after_reprocess(auth_client, monkeypatch):
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Content about reinforcement learning agents.")
    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("cache_invalidation_test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_response = auth_client.post("/api/v1/documents/upload", files=files)
    document_id = upload_response.json()["results"][0]["document_id"]
    time.sleep(0.5)

    call_count = {"n": 0}
    from app.rag import gemini_client as gemini_module

    def counting_generate(self, prompt, temperature=0.0):
        call_count["n"] += 1
        return "mocked summary text"

    monkeypatch.setattr(gemini_module.GeminiClient, "generate", counting_generate)

    auth_client.post("/api/v1/analysis/summarize", json={"document_id": document_id})
    calls_after_first = call_count["n"]

    # Reprocessing must invalidate the cached summary.
    reprocess_response = auth_client.post(f"/api/v1/documents/{document_id}/reprocess")
    assert reprocess_response.status_code == 200
    time.sleep(0.5)

    auth_client.post("/api/v1/analysis/summarize", json={"document_id": document_id})
    assert call_count["n"] > calls_after_first  # a fresh Gemini call happened after invalidation
