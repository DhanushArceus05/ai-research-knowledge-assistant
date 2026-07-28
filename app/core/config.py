"""
Application configuration, loaded from environment variables using Pydantic Settings.
"""
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings. Values are read from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_NAME: str = "AI Research & Knowledge Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "sqlite:///./data/app.db"

    # Storage
    UPLOAD_DIR: str = "./data/raw_documents"
    VECTOR_DB_DIR: str = "./data/vector_db"

    # ML model artifacts
    MODEL_PATH: str = "./models/document_classifier.keras"
    VECTORIZER_PATH: str = "./models/text_vectorizer.pkl"
    LABEL_ENCODER_PATH: str = "./models/label_encoder.pkl"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking / retrieval
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 4
    MAX_UPLOAD_SIZE_MB: int = 20

    # Conversation memory
    MAX_HISTORY_MESSAGES: int = 8

    # Authentication / JWT
    JWT_SECRET_KEY: str = "dev-only-change-me-please"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # OCR
    OCR_ENABLED: bool = True
    OCR_TEXT_THRESHOLD: int = 40
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    TESSERACT_CMD: str = ""

    # Reranking
    RERANKING_ENABLED: bool = True
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_CANDIDATE_COUNT: int = 12

    # Caching
    CACHE_BACKEND: str = "memory"  # "memory" or "redis"
    CACHE_TTL_SECONDS: int = 300
    REDIS_URL: str = "redis://redis:6379/0"

    # Agent
    AGENT_MAX_STEPS: int = 5

    # Streaming
    STREAMING_ENABLED: bool = True

    # Hybrid retrieval defaults
    HYBRID_DENSE_CANDIDATES: int = 15
    HYBRID_SPARSE_CANDIDATES: int = 15
    RRF_K: int = 60

    # Multi-modal extraction
    IMAGES_DIR: str = "./data/extracted_images"

    def ensure_directories(self) -> None:
        """Create all required directories if they do not already exist."""
        for path in (
            self.UPLOAD_DIR, self.VECTOR_DB_DIR, Path(self.MODEL_PATH).parent,
            "./data/dataset", self.IMAGES_DIR,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env file is only parsed once."""
    settings = Settings()
    settings.ensure_directories()
    return settings
