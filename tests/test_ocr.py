from app.document_processing.ocr import OCRService


def _make_page(text_via_fitz: str):
    fitz = __import__("fitz")
    doc = fitz.open()
    page = doc.new_page()
    if text_via_fitz:
        page.insert_text((72, 72), text_via_fitz)
    return doc, page


def test_needs_ocr_false_for_normal_text_page():
    ocr = OCRService()
    normal_text = "This is a perfectly normal page with plenty of extractable text content."
    assert ocr.needs_ocr(normal_text) is False


def test_needs_ocr_true_for_sparse_or_empty_text():
    ocr = OCRService()
    assert ocr.needs_ocr("") is True
    assert ocr.needs_ocr("hi") is True


def test_needs_ocr_respects_disabled_config():
    ocr = OCRService()
    ocr.enabled_by_config = False
    assert ocr.needs_ocr("") is False


def test_ocr_unavailable_returns_none_without_crashing(monkeypatch):
    """Simulates Tesseract not being installed/discoverable at all."""
    ocr = OCRService()
    monkeypatch.setattr(ocr, "_detect_tesseract", lambda: False)

    doc, page = _make_page("")
    try:
        result = ocr.extract_text_from_page_image(page)
        assert result is None
    finally:
        doc.close()


def test_ocr_extracts_text_when_available(monkeypatch):
    """Mocks pytesseract's actual OCR call so this test is deterministic and fast,
    while still exercising the real page-rendering code path via PyMuPDF."""
    ocr = OCRService()
    monkeypatch.setattr(ocr, "_detect_tesseract", lambda: True)
    monkeypatch.setattr(ocr, "is_available", lambda: True)

    import pytesseract
    monkeypatch.setattr(pytesseract, "image_to_string", lambda img, lang=None: "mocked ocr text output")

    doc, page = _make_page("")
    try:
        result = ocr.extract_text_from_page_image(page)
        assert result == "mocked ocr text output"
    finally:
        doc.close()


def test_ocr_handles_extraction_exception_gracefully(monkeypatch):
    ocr = OCRService()
    monkeypatch.setattr(ocr, "is_available", lambda: True)

    import pytesseract

    def raise_error(*args, **kwargs):
        raise RuntimeError("simulated OCR engine crash")

    monkeypatch.setattr(pytesseract, "image_to_string", raise_error)

    doc, page = _make_page("")
    try:
        result = ocr.extract_text_from_page_image(page)
        assert result is None  # gracefully returns None instead of raising
    finally:
        doc.close()


def test_tesseract_detection_against_installed_binary():
    """Exercises the real shutil.which() / pytesseract.get_tesseract_version()
    path against whatever Tesseract install is actually present, rather than
    mocking detection out."""
    ocr = OCRService()
    ocr.enabled_by_config = True
    available = ocr.is_available()
    assert available is True
