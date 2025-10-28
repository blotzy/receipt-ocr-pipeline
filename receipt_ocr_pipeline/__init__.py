"""
Receipt OCR Pipeline

A local, scriptable pipeline to process receipt scans into structured data
and submission-ready PDF packages.
"""

__version__ = "1.0.0"
__author__ = "Receipt OCR Pipeline Contributors"

from receipt_ocr_pipeline.core.models import ReceiptEntry

__all__ = ["ReceiptEntry"]
