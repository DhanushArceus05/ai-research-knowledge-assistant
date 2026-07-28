"""
OCR fallback for scanned/low-text PDF pages.

Uses PyMuPDF to render a page to an image and pytesseract (a thin wrapper
around the Tesseract binary) to extract text from it. If the Tesseract binary
is not installed/discoverable, OCR is cleanly disabled: normal text-based PDF
processing is completely unaffected, and the affected pages are reported as
"OCR unavailable" rather than causing a crash.
"""
import shutil
from functools import lru_cache
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OCRService:
    def __init__(self):
        settings = get_settings()
        self.enabled_by_config = settings.OCR_ENABLED
        self.language = settings.OCR_LANGUAGE
        self.dpi = settings.OCR_DPI
        self.text_threshold = settings.OCR_TEXT_THRESHOLD
        self._tesseract_checked = False
        self._tesseract_available = False

        if settings.TESSERACT_CMD:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

    def _detect_tesseract(self) -> bool:
        if self._tesseract_checked:
            return self._tesseract_available
        self._tesseract_checked = True

        try:
            import pytesseract
            configured_cmd = pytesseract.pytesseract.tesseract_cmd
            found = configured_cmd and configured_cmd != "tesseract" and shutil.which(configured_cmd)
            found = found or shutil.which("tesseract")
            if not found:
                logger.warning(
                    "Tesseract not found on PATH and TESSERACT_CMD is unset; "
                    "OCR will be skipped for low-text pages."
                )
                self._tesseract_available = False
                return False

            # Confirm pytesseract can actually invoke the binary.
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            return True
        except Exception:
            logger.warning("Tesseract detection failed; OCR will be skipped.", exc_info=True)
            self._tesseract_available = False
            return False

    def is_available(self) -> bool:
        return self.enabled_by_config and self._detect_tesseract()

    def needs_ocr(self, extracted_text: str) -> bool:
        """Returns True if a page's normally-extracted text is too sparse and OCR should be attempted."""
        return self.enabled_by_config and len(extracted_text.strip()) < self.text_threshold

    def extract_text_from_page_image(self, page, dpi: Optional[int] = None) -> Optional[str]:
        """Renders a PyMuPDF page to an image and runs OCR on it. Returns None if OCR is unavailable."""
        if not self.is_available():
            return None

        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=dpi or self.dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=self.language)
            return text.strip()
        except Exception:
            logger.warning("OCR extraction failed for a page; continuing without OCR text for it.", exc_info=True)
            return None


@lru_cache
def get_ocr_service() -> OCRService:
    return OCRService()
