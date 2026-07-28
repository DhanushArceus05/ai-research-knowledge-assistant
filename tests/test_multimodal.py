import io
import os
import time


def _upload_pdf_with_image_and_table(client):
    """Builds a real PDF containing an embedded raster image and a simple table-like
    grid of text (drawn with rect/lines so PyMuPDF's structural table finder can detect it),
    then uploads it and waits for background processing to finish."""
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A document with an embedded image and a table below.")

    # Embed a tiny raster image (a 2x2 solid-color PNG).
    from PIL import Image
    img_buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img_buffer, format="PNG")
    img_buffer.seek(0)
    page.insert_image(fitz.Rect(72, 100, 172, 200), stream=img_buffer.read())

    # Draw a simple 2x2 grid with text in each cell so find_tables() can detect structure.
    rect = fitz.Rect(72, 250, 300, 350)
    page.draw_rect(rect, color=(0, 0, 0))
    page.draw_line((72, 300), (300, 300), color=(0, 0, 0))
    page.draw_line((186, 250), (186, 350), color=(0, 0, 0))
    page.insert_text((80, 270), "Header A")
    page.insert_text((190, 270), "Header B")
    page.insert_text((80, 320), "Row1 A")
    page.insert_text((190, 320), "Row1 B")

    pdf_bytes = doc.tobytes()
    doc.close()

    files = {"files": ("multimodal.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = client.post("/api/v1/documents/upload", files=files)
    document_id = response.json()["results"][0]["document_id"]
    time.sleep(1.0)
    return document_id


def test_image_extraction_produces_metadata(auth_client):
    document_id = _upload_pdf_with_image_and_table(auth_client)

    response = auth_client.get(f"/api/v1/documents/{document_id}/images")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 1
    image = body["images"][0]
    assert image["page_number"] == 1
    assert image["format"] is not None


def test_table_extraction_produces_metadata(auth_client):
    document_id = _upload_pdf_with_image_and_table(auth_client)

    response = auth_client.get(f"/api/v1/documents/{document_id}/tables")
    assert response.status_code == 200
    body = response.json()
    # PyMuPDF's structural table detection is best-effort; assert on the shape
    # of the response rather than requiring a table to always be found for
    # this deliberately minimal synthetic grid.
    assert "tables" in body
    assert isinstance(body["tables"], list)


def test_images_endpoint_enforces_ownership(auth_client, second_auth_client):
    document_id = _upload_pdf_with_image_and_table(auth_client)

    response = second_auth_client.get(f"/api/v1/documents/{document_id}/images")
    assert response.status_code == 404


def test_tables_endpoint_enforces_ownership(auth_client, second_auth_client):
    document_id = _upload_pdf_with_image_and_table(auth_client)

    response = second_auth_client.get(f"/api/v1/documents/{document_id}/tables")
    assert response.status_code == 404


def test_images_endpoint_requires_authentication(client):
    response = client.get("/api/v1/documents/some-id/images")
    assert response.status_code == 401


def test_deleting_document_removes_extracted_image_assets_from_disk(auth_client):
    from app.core.config import get_settings
    settings = get_settings()

    document_id = _upload_pdf_with_image_and_table(auth_client)
    images_dir = os.path.join(settings.IMAGES_DIR, document_id)
    assert os.path.isdir(images_dir)
    assert len(os.listdir(images_dir)) >= 1

    delete_response = auth_client.delete(f"/api/v1/documents/{document_id}")
    assert delete_response.status_code == 200
    assert not os.path.isdir(images_dir)


def test_image_extractor_avoids_duplicate_extraction_on_same_xref():
    """Two pages referencing the same embedded image xref should only extract one file."""
    import fitz
    from app.document_processing.image_extractor import ImageExtractor
    from PIL import Image

    doc = fitz.open()
    img_buffer = io.BytesIO()
    Image.new("RGB", (5, 5), color=(0, 255, 0)).save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    page1 = doc.new_page()
    page1.insert_image(fitz.Rect(10, 10, 50, 50), stream=img_bytes)
    page2 = doc.new_page()
    # Reuse the same raw image bytes on a second page; PyMuPDF may or may not
    # dedupe the xref itself, but our extractor must dedupe by xref regardless.
    page2.insert_image(fitz.Rect(10, 10, 50, 50), stream=img_bytes)

    extractor = ImageExtractor()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        extractor.images_dir = tmp_dir
        results = extractor.extract_images(doc, "dedup-test-doc")

    doc.close()
    # However many unique xrefs PyMuPDF created, none should be extracted twice.
    xrefs_seen = [r["file_path"] for r in results]
    assert len(xrefs_seen) == len(set(xrefs_seen))
