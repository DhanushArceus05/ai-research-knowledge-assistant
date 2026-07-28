"""
Extracts embedded images from PDF pages using PyMuPDF, saving them to a
per-document directory and returning metadata rows ready for persistence.
"""
import os
from typing import List, Dict, Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ImageExtractor:
    def __init__(self):
        settings = get_settings()
        self.images_dir = settings.IMAGES_DIR

    def extract_images(self, doc, document_id: str) -> List[Dict[str, Any]]:
        """
        Extracts every embedded raster image from a PyMuPDF document. `doc` is an
        already-open `fitz.Document`. Returns a list of dicts ready to become
        ImageAsset rows (without user_id/document_id, which the caller fills in).
        Does not attempt duplicate extraction: called once per (re)processing run
        after any previous extracted-image directory for this document was removed.
        """
        output_dir = os.path.join(self.images_dir, document_id)
        os.makedirs(output_dir, exist_ok=True)

        results: List[Dict[str, Any]] = []
        seen_xrefs = set()

        for page_index in range(doc.page_count):
            page = doc[page_index]
            for image_index, image_info in enumerate(page.get_images(full=True)):
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue  # the same image object may be referenced by multiple pages
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    logger.warning("Failed to extract image xref=%s on page %d.", xref, page_index + 1)
                    continue

                image_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")
                if not image_bytes:
                    continue

                file_name = f"page{page_index + 1}_img{image_index}_{xref}.{ext}"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                results.append({
                    "page_number": page_index + 1,
                    "file_path": file_path,
                    "width": base_image.get("width"),
                    "height": base_image.get("height"),
                    "format": ext,
                })

        return results
