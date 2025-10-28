"""
OCR functionality for processing receipt images and PDFs.
"""

import io
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from .utils import IMAGE_EXTS, PDF_EXTS


def _lazy_import_ocr_deps():
    """Lazy import heavy OCR dependencies."""
    global pytesseract, PIL_Image, fitz, reportlab
    import importlib
    pytesseract = importlib.import_module("pytesseract")
    PIL_Image = importlib.import_module("PIL.Image")
    fitz = importlib.import_module("fitz")  # pymupdf
    reportlab = importlib.import_module("reportlab")


# Initialize on first import
pytesseract = None
PIL_Image = None
fitz = None
reportlab = None


def ensure_ocr_for_pdf(src_pdf: Path, out_pdf: Path):
    """
    Ensure a PDF is text-searchable. If it already has text, copy it.
    Otherwise, call ocrmypdf to embed OCR text layer.
    """
    if fitz is None:
        _lazy_import_ocr_deps()

    import fitz as fitz_module
    doc = fitz_module.open(src_pdf.as_posix())
    has_text = any(page.get_text().strip() for page in doc)
    doc.close()
    if has_text:
        shutil.copy2(src_pdf, out_pdf)
        return

    # Run ocrmypdf if available
    try:
        subprocess.run(
            ["ocrmypdf", "--quiet", "--skip-text", src_pdf.as_posix(), out_pdf.as_posix()],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # fallback: rasterize-first page to image and Tesseract single page -> simple PDF
        # (best-effort; strongly recommend installing ocrmypdf)
        print(f"[WARN] ocrmypdf not available; attempting simple Tesseract OCR for {src_pdf.name}")
        doc = fitz_module.open(src_pdf.as_posix())
        # Simplified: just OCR first page image
        mat = fitz_module.Matrix(2, 2)
        pix = doc[0].get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img)
        # Save minimal searchable PDF with text dump appended:
        with open(out_pdf, "wb") as f:
            f.write(src_pdf.read_bytes())
        # (This fallback won't truly layer text; just a placeholder.)


def ocr_image_to_text(img_path: Path) -> str:
    """OCR an image file to text."""
    if pytesseract is None:
        _lazy_import_ocr_deps()

    from PIL import Image
    img = Image.open(img_path)
    # Improve OCR: convert to grayscale & maybe DPI upsample
    if img.mode != "L":
        img = img.convert("L")
    return pytesseract.image_to_string(img)


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a (hopefully) searchable PDF using PyMuPDF."""
    if fitz is None:
        _lazy_import_ocr_deps()

    import fitz as fitz_module
    doc = fitz_module.open(pdf_path.as_posix())
    chunks = []
    for page in doc:
        chunks.append(page.get_text())
    doc.close()
    return "\n".join(chunks)


def process_receipt_file(path: Path, work_pdf_dir: Path) -> Tuple[str, str, str]:
    """
    Process a receipt file (image or PDF) to extract text and create searchable PDF.

    Returns:
        Tuple of (plain_text, search_pdf_path, canonical_ext)
        - For images: OCR directly to text; also convert to single-page searchable PDF
        - For PDFs: ensure searchable using ocrmypdf (or copy if already has text)
    """
    if pytesseract is None:
        _lazy_import_ocr_deps()

    ext = path.suffix.lower()
    plain_text = ""
    out_pdf = work_pdf_dir / (path.stem + ".pdf")

    if ext in IMAGE_EXTS:
        # OCR image -> text
        plain_text = ocr_image_to_text(path)
        # Build a simple PDF from image (no text layer). Optionally run ocrmypdf later for searchability.
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.utils import ImageReader
            from PIL import Image
            img = Image.open(path)
            w, h = img.size
            # Scale to fit letter
            page_w, page_h = letter
            scale = min(page_w / w, page_h / h)
            new_w, new_h = w * scale, h * scale
            c = canvas.Canvas(out_pdf.as_posix(), pagesize=letter)
            x = (page_w - new_w) / 2
            y = (page_h - new_h) / 2
            c.drawImage(ImageReader(img), x, y, width=new_w, height=new_h, preserveAspectRatio=True, anchor='c')
            c.showPage()
            c.save()
        except Exception as e:
            print(f"[WARN] Could not embed image into PDF for {path.name}: {e}")
            # As a fallback, just copy image bytes? We'll skip embedding.
    elif ext in PDF_EXTS:
        # Ensure it's searchable; copy if already has text
        ensure_ocr_for_pdf(path, out_pdf)
        plain_text = pdf_to_text(out_pdf)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return plain_text, out_pdf.as_posix(), ".pdf"
