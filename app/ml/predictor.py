"""
Inference wrapper for the TensorFlow document classifier.

The model is loaded lazily. If the trained artifacts are not present (e.g. on a
fresh checkout before `python scripts/train_classifier.py` has been run), the
predictor returns an explicit UNCLASSIFIED result instead of crashing the API.
"""
import os
import pickle
from typing import Tuple
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

UNCLASSIFIED = "UNCLASSIFIED"


class DocumentClassifier:
    def __init__(self):
        settings = get_settings()
        self.model_path = settings.MODEL_PATH
        self.label_encoder_path = settings.LABEL_ENCODER_PATH
        self._model = None
        self._label_encoder = None
        self._load_attempted = False

    def _available(self) -> bool:
        return os.path.exists(self.model_path) and os.path.exists(self.label_encoder_path)

    def _load(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True

        if not self._available():
            logger.warning(
                "Classifier model artifacts not found at '%s'. "
                "Run `python scripts/train_classifier.py` to train the model. "
                "Uploads will be classified as '%s' until then.",
                self.model_path, UNCLASSIFIED,
            )
            return False

        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self.model_path)
            with open(self.label_encoder_path, "rb") as f:
                self._label_encoder = pickle.load(f)
            logger.info("Loaded document classifier from %s", self.model_path)
            return True
        except Exception:
            logger.exception("Failed to load document classifier; falling back to UNCLASSIFIED.")
            self._model = None
            return False

    def predict(self, text: str) -> Tuple[str, float]:
        """Returns (category, confidence). Falls back to (UNCLASSIFIED, 0.0) on any issue."""
        if not text or not text.strip():
            return UNCLASSIFIED, 0.0

        if not self._load():
            return UNCLASSIFIED, 0.0

        try:
            import tensorflow as tf
            # Use a representative excerpt to keep inference fast for very long documents.
            excerpt = text[:5000]
            probabilities = self._model.predict(tf.constant([excerpt], dtype=tf.string), verbose=0)[0]
            best_index = int(probabilities.argmax())
            category = self._label_encoder.inverse_transform([best_index])[0]
            confidence = float(probabilities[best_index])
            return str(category), confidence
        except Exception:
            logger.exception("Classification inference failed; falling back to UNCLASSIFIED.")
            return UNCLASSIFIED, 0.0


@lru_cache
def get_document_classifier() -> DocumentClassifier:
    return DocumentClassifier()
