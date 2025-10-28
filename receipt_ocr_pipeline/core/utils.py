"""
Utility functions and constants for receipt processing.
"""

import hashlib
import re
import datetime as dt
from pathlib import Path
from typing import Optional

# File type constants
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}

# Pattern constants for parsing
DATE_PATTERNS = [
    r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",            # YYYY-MM-DD or YYYY/MM/DD
    r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b",          # MM/DD/YYYY or DD/MM/YYYY (heuristic later)
    r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b",       # Month DD, YYYY
]

AMOUNT_PATTERNS = [
    # Labeled amounts (most reliable)
    r"(?:total|amount|balance|due|pay|grand\s*total|sum)\s*[:\-=]?\s*\$?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?)",
    r"(?:total|amount|balance|due|pay)\s*\$?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?)",

    # Currency symbol patterns
    r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))",  # Require decimal for $ amounts
    r"USD\s*\$?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?)",

    # End of line amounts (common in receipts)
    r"\s([0-9]{1,3}(?:[,\s][0-9]{3})*\.[0-9]{2})\s*$",

    # Standalone dollar amounts with decimals
    r"\b([0-9]{1,3}(?:,[0-9]{3})+\.[0-9]{2})\b",  # 1,234.56
    r"\b([0-9]{2,3}\.[0-9]{2})\b",  # 12.34 or 123.45
]

VENDOR_PATTERN = r"^[^\n]{2,80}$"


def slugify(s: str) -> str:
    """Convert string to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def normalize_amount(s: str) -> Optional[float]:
    """Normalize amount string to float."""
    if not s:
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None


def sha1_file(path: Path) -> str:
    """Calculate SHA1 hash of file."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_current_week() -> str:
    """Return current week in ISO format: YYYY-Www (e.g., 2025-W43)."""
    return dt.date.today().strftime("%Y-W%U")


def compute_receipt_fingerprint(date: Optional[str], vendor: Optional[str], amount: Optional[float]) -> str:
    """Create a fingerprint for duplicate detection based on date, vendor, and amount."""
    parts = [
        date or "",
        (vendor or "").strip().lower(),
        f"{amount:.2f}" if amount is not None else ""
    ]
    fingerprint_str = "|".join(parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()


def money_fmt(v: Optional[float]) -> str:
    """Format amount as currency."""
    return f"${v:,.2f}" if v is not None else ""
