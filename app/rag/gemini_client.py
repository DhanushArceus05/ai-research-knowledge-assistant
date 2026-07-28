"""
Thin wrapper around the Google Gemini API used by all RAG services.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured. Set it in your .env file to use AI features."
            )
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)
        return self._client

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        client = self._get_client()
        try:
            response = client.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            return (response.text or "").strip()
        except Exception:
            logger.exception("Gemini generation failed.")
            raise ConfigurationError("The AI generation service failed to respond. Please try again.")

    def generate_stream(self, prompt: str, temperature: float = 0.0):
        """
        Yields incremental text chunks using the Gemini SDK's native streaming
        support (`stream=True`). Raises ConfigurationError on failure so the
        caller can emit a structured error event instead of crashing mid-stream.
        """
        client = self._get_client()
        try:
            response = client.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception:
            logger.exception("Gemini streaming generation failed.")
            raise ConfigurationError("The AI generation service failed to respond while streaming.")


@lru_cache
def get_gemini_client() -> GeminiClient:
    return GeminiClient()
