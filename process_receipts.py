#!/usr/bin/env python3
"""
process_receipts.py
Local, scriptable pipeline to turn a folder of receipt scans into:
- receipts.csv (structured data)
- summary.pdf (category totals + line-items)
- claim_package.pdf (summary followed by all original receipts)
- receipts.sqlite (optional) with a simple table of parsed entries

Weekly Batch Processing:
Organizes receipts by week with separate directories for each batch.
Detects duplicates across all weeks (same date, vendor, amount).

Workflow:
1) Place new receipt images/PDFs into <incoming_dir>
2) Run:
   python process_receipts.py --incoming ./incoming --weeks ./weeks --rules ./rules.json [--week 2025-W43]
3) Script OCRs, extracts date/vendor/amount, categorizes using rules, and bundles a submission PDF
4) Each week's data is stored in weeks/<week_id>/{processed,output}/

Requires: Python 3.9+, Tesseract OCR (binary), plus Python packages in requirements.txt
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# --- Optional heavy deps (imported lazily) ---
def _lazy_imports():
    global pytesseract, PIL_Image, fitz, reportlab, pypdf
    import importlib
    pytesseract = importlib.import_module("pytesseract")
    PIL_Image = importlib.import_module("PIL.Image")
    fitz = importlib.import_module("fitz")  # pymupdf
    reportlab = importlib.import_module("reportlab")
    pypdf = importlib.import_module("pypdf")

# --- LLM client (imported lazily) ---
_anthropic_client = None

def _get_anthropic_client():
    """Get or create Anthropic client (lazy initialization)"""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    return _anthropic_client

# --------------- Utilities ---------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}

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

# A simple vendor heuristic: first ALLCAPS token with letters, or top line fallback
VENDOR_PATTERN = r"^[^\n]{2,80}$"

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def normalize_amount(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None

def parse_date(text: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            g = m.groups()
            # Try to interpret
            try:
                if len(g) == 3:
                    # pattern-based normalization
                    if pat.startswith(r"\b(\d{4})"):
                        y, mo, d = int(g[0]), int(g[1]), int(g[2])
                    elif pat.startswith(r"\b(\d{1,2})"):
                        # Heuristic: if last group is 4 digits -> year
                        mo, d, y = int(g[0]), int(g[1]), int(g[2])
                        if y < 100:  # YY -> 20YY
                            y += 2000
                        # If looks like DD/MM, swap if mo > 12
                        if mo > 12 and d <= 12:
                            mo, d = d, mo
                    else:
                        # Month name
                        mn, d, y = g[0], int(g[1]), int(g[2])
                        mo = dt.datetime.strptime(mn[:3], "%b").month
                    dtobj = dt.date(y, mo, d)
                    return dtobj.isoformat()
            except Exception:
                continue
    return None

def parse_amount(text: str) -> Optional[float]:
    """
    Extract the total amount from receipt text.
    Tries multiple patterns, prioritizing labeled amounts.
    """
    candidates = []

    for pat in AMOUNT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            val = normalize_amount(m.group(1))
            if val and val > 0:
                # Track position to prioritize amounts near end of receipt
                candidates.append((val, m.start()))

    if not candidates:
        return None

    # If we have labeled amounts (from first 2 patterns), prefer those
    labeled_amounts = [c for c in candidates[:5] if c[0]]  # First few are from labeled patterns
    if labeled_amounts:
        # Return the largest labeled amount (usually the total)
        return max(labeled_amounts, key=lambda x: x[0])[0]

    # Otherwise, prefer amounts that appear later in the text (near bottom of receipt)
    # and are larger (likely the total rather than subtotals)
    if candidates:
        # Sort by position (later is better) and value (larger is better)
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return candidates[0][0]

    return None

# --------------- LLM-based Extraction ---------------

def init_llm_cache_db(db_path: Path):
    """Initialize SQLite database for LLM extraction cache"""
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS llm_cache (
        file_hash TEXT PRIMARY KEY,
        vendor TEXT,
        date TEXT,
        category TEXT,
        confidence REAL,
        reasoning TEXT,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def get_llm_cache(db_path: Path, file_hash: str) -> Optional[Dict]:
    """Retrieve cached LLM result from SQLite"""
    try:
        conn = sqlite3.connect(db_path.as_posix())
        cur = conn.cursor()
        cur.execute("""
            SELECT vendor, date, category, confidence, reasoning
            FROM llm_cache
            WHERE file_hash = ?
        """, (file_hash,))
        result = cur.fetchone()
        conn.close()

        if result:
            return {
                "vendor": result[0],
                "date": result[1],
                "category": result[2],
                "confidence": result[3],
                "reasoning": result[4],
                "cached": True
            }
        return None
    except Exception as e:
        print(f"[WARN] Could not read from LLM cache: {e}")
        return None

def save_llm_cache(db_path: Path, file_hash: str, vendor: Optional[str], date: Optional[str],
                   category: str, confidence: float, reasoning: str):
    """Save LLM extraction result to SQLite cache"""
    try:
        conn = sqlite3.connect(db_path.as_posix())
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO llm_cache
            (file_hash, vendor, date, category, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_hash, vendor, date, category, confidence, reasoning))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARN] Could not save to LLM cache: {e}")

def extract_with_llm(ocr_text: str, categories: List[str], use_llm: bool = True, file_hash: Optional[str] = None, cache_path: Optional[Path] = None) -> Dict:
    """
    Extract vendor name and category from receipt text using Claude.

    Args:
        ocr_text: Full OCR text from receipt
        categories: List of available categories
        use_llm: Whether to use LLM (can be disabled for testing/fallback)
        file_hash: SHA1 hash of file for caching (optional)
        cache_path: Path to cache file (optional)

    Returns:
        Dict with keys: vendor, category, confidence (0.0-1.0), reasoning, cached (bool)
    """
    if not use_llm:
        return {"vendor": None, "category": "Uncategorized", "confidence": 0.0, "reasoning": "LLM disabled", "cached": False}

    # Check cache first
    if file_hash and cache_path:
        cached_result = get_llm_cache(cache_path, file_hash)
        if cached_result:
            return cached_result

    try:
        client = _get_anthropic_client()

        # Truncate text to avoid token limits (keep first ~2000 chars, usually enough)
        text_sample = ocr_text[:2000]

        prompt = f"""Analyze this receipt text and extract information.

Receipt text (OCR output, may contain errors):
{text_sample}

Tasks:
1. Identify the VENDOR/BUSINESS name (the company that issued this receipt)
2. Find the TRANSACTION/ORDER DATE (the date of purchase, NOT print date or payment date)
3. Classify into ONE of these categories: {', '.join(categories)}

Guidelines:
- Vendor is usually at the top of the receipt
- Ignore customer names, cardholder names, or "sold to" fields
- For hotels/inns: extract the full property name (e.g., "Inn at Thorn Hill", "Cambria Hotel")
- For Airbnb: use "Airbnb" as vendor
- Date: Look for "Order Date", "Transaction Date", "Purchase Date", or similar
  * Ignore "Print Date", "Payment Date", or today's date if it's clearly a reprint
  * Return in YYYY-MM-DD format
- Categories:
  * Accommodations: hotels, motels, inns, lodges, resorts, Airbnb, vacation rentals
  * Food/Grocery: restaurants, cafes, grocery stores, supermarkets
  * Other categories as listed

Return ONLY a JSON object (no markdown, no explanation):
{{
  "vendor": "Business Name Here",
  "date": "2025-10-27",
  "category": "Category Name",
  "confidence": 0.95,
  "reasoning": "Brief explanation of your decision"
}}"""

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast and cost-effective
            max_tokens=300,
            temperature=0.0,  # Deterministic
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        response_text = response.content[0].text.strip()

        # Handle markdown code blocks if present
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = []
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        result = json.loads(response_text)

        # Validate and normalize
        vendor = result.get("vendor", "").strip()
        date = result.get("date", "").strip()
        category = result.get("category", "Uncategorized").strip()
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        # Ensure category is valid
        if category not in categories:
            # Try case-insensitive match
            category_lower = category.lower()
            matched = [c for c in categories if c.lower() == category_lower]
            if matched:
                category = matched[0]
            else:
                category = "Uncategorized"

        result = {
            "vendor": vendor if vendor else None,
            "date": date if date else None,
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
            "cached": False
        }

        # Save to cache only if confidence is good (worth caching)
        # Don't cache low-confidence results that will trigger fallback anyway
        if file_hash and cache_path and confidence >= 0.3:
            save_llm_cache(cache_path, file_hash, vendor, date, category, confidence, reasoning)

        return result

    except Exception as e:
        # Fallback on error
        return {
            "vendor": None,
            "date": None,
            "category": "Uncategorized",
            "confidence": 0.0,
            "reasoning": f"LLM extraction failed: {str(e)}",
            "cached": False
        }

def parse_vendor(text: str, blacklist: Optional[List[str]] = None, hints: Optional[Dict[str, str]] = None, rules: Optional[Dict] = None) -> Optional[str]:
    """
    Extract vendor name from receipt text with improved heuristics.

    Args:
        text: OCR'd receipt text
        blacklist: List of strings to exclude (e.g., customer names)
        hints: Dict mapping text patterns to vendor names for manual override
        rules: Full rules dict with matchers that may contain vendor_extract patterns

    Returns:
        Vendor name or None
    """
    blacklist = blacklist or []
    blacklist_lower = [b.lower() for b in blacklist]
    hints = hints or {}
    rules = rules or {}

    # First, check vendor hints for manual overrides
    text_lower = text.lower()
    for pattern, vendor_name in hints.items():
        if pattern.lower() in text_lower:
            return vendor_name

    # Try to extract using patterns from matchers
    for matcher in rules.get("matchers", []):
        vendor_extract_patterns = matcher.get("vendor_extract", [])
        for pattern_obj in vendor_extract_patterns:
            pattern = pattern_obj.get("pattern")
            if pattern:
                match = re.search(pattern, text)
                if match:
                    extracted = match.group(0)
                    # Check if it's not in blacklist
                    if not any(bl in extracted.lower() for bl in blacklist_lower):
                        return extracted

    # Common patterns to skip (customer-related text, receipt metadata)
    skip_patterns = [
        r'^(customer|cardholder|card\s*holder|member|account|name)[\s:]*',
        r'^(thank\s*you|thanks|receipt|invoice|bill|order|ticket)',
        r'^(date|time|server|cashier|clerk|sold\s*to|ship\s*to)',
        r'^\*+\s*$',  # Lines that are just asterisks
        r'^-+\s*$',   # Lines that are just dashes
        r'^=+\s*$',   # Lines that are just equals
        r'^[0-9\s\-\./]+$',  # Lines with only numbers and punctuation
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates at start of line
        r'^(visa|mastercard|amex|discover|debit|credit)',  # Card types
    ]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = []
    for idx, ln in enumerate(lines[:20]):  # Check first 20 lines
        if len(ln) < 3 or len(ln) > 80:
            continue

        # Skip if matches any skip pattern
        skip = False
        for pattern in skip_patterns:
            if re.search(pattern, ln, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        # Skip if in blacklist
        if any(bl in ln.lower() for bl in blacklist_lower):
            continue

        letters = sum(c.isalpha() for c in ln)
        digits = sum(c.isdigit() for c in ln)
        spaces = ln.count(' ')

        # Must have mostly letters, not too many digits or spaces
        if letters >= 3 and digits < max(2, letters // 3) and spaces < 5:
            # Boost score for lines near the top
            score = 100 - idx * 3  # Earlier lines get higher scores

            # Boost for all-caps (common for business names)
            if ln.upper() == ln and 3 <= len(ln) <= 40:
                score += 50

            # Boost for moderate length (sweet spot for business names)
            if 5 <= len(ln) <= 30:
                score += 20

            # Penalty for very long lines
            if len(ln) > 50:
                score -= 30

            # Penalty for mixed case with lots of spaces (likely address or description)
            if spaces > 2:
                score -= 15

            candidates.append((score, ln))

    if candidates:
        # Sort by score and return the best candidate
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]

    # Last resort: return first non-blacklisted line
    for ln in lines[:5]:
        if len(ln) >= 3 and not any(bl in ln.lower() for bl in blacklist_lower):
            skip = False
            for pattern in skip_patterns:
                if re.search(pattern, ln, re.IGNORECASE):
                    skip = True
                    break
            if not skip:
                return ln

    return lines[0] if lines else None

def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_current_week() -> str:
    """Return current week in ISO format: YYYY-Www (e.g., 2025-W43)"""
    return dt.date.today().strftime("%Y-W%U")

def compute_receipt_fingerprint(date: Optional[str], vendor: Optional[str], amount: Optional[float]) -> str:
    """Create a fingerprint for duplicate detection based on date, vendor, and amount"""
    parts = [
        date or "",
        (vendor or "").strip().lower(),
        f"{amount:.2f}" if amount is not None else ""
    ]
    fingerprint_str = "|".join(parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()

# --------------- OCR ---------------

def ensure_ocr_for_pdf(src_pdf: Path, out_pdf: Path):
    """
    Ensure a PDF is text-searchable. If it already has text, copy it.
    Otherwise, call ocrmypdf to embed OCR text layer.
    """
    import fitz  # lazy ok
    doc = fitz.open(src_pdf.as_posix())
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
        import fitz
        doc = fitz.open(src_pdf.as_posix())
        # Simplified: just OCR first page image
        mat = fitz.Matrix(2, 2)
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
    from PIL import Image
    img = Image.open(img_path)
    # Improve OCR: convert to grayscale & maybe DPI upsample
    if img.mode != "L":
        img = img.convert("L")
    return pytesseract.image_to_string(img)

def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a (hopefully) searchable PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(pdf_path.as_posix())
    chunks = []
    for page in doc:
        chunks.append(page.get_text())
    doc.close()
    return "\n".join(chunks)

# --------------- Categorization ---------------

def load_rules(path: Path) -> Dict:
    if not path.exists():
        return {"categories": [], "matchers": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def categorize(vendor: str, text: str, rules: Dict) -> Tuple[str, Optional[str]]:
    """
    Returns (category, matcher_name)
    Rules format:
    {
      "categories": ["Hotel","Meals","Transportation","Misc"],
      "matchers": [
        {"name":"Hotels","any":[{"vendor_re":"HILTON|MARRIOTT|HYATT"}], "category":"Hotel"},
        {"name":"Uber/Lyft","any":[{"text_re":"UBER|LYFT"}], "category":"Transportation"}
      ]
    }
    """
    v = vendor or ""
    t = text or ""
    for m in rules.get("matchers", []):
        any_rules = m.get("any", [])
        all_rules = m.get("all", [])
        matched_any = False
        if any_rules:
            for rule in any_rules:
                vre = rule.get("vendor_re")
                tre = rule.get("text_re")
                ok = False
                if vre and re.search(vre, v, flags=re.IGNORECASE):
                    ok = True
                if tre and re.search(tre, t, flags=re.IGNORECASE):
                    ok = True
                if ok:
                    matched_any = True
                    break
        else:
            matched_any = True  # no 'any' means don't gate on it
        matched_all = True
        for rule in all_rules:
            vre = rule.get("vendor_re")
            tre = rule.get("text_re")
            ok = True
            if vre and not re.search(vre, v, flags=re.IGNORECASE):
                ok = False
            if tre and not re.search(tre, t, flags=re.IGNORECASE):
                ok = False
            if not ok:
                matched_all = False
                break
        if matched_any and matched_all:
            return (m.get("category") or "Uncategorized", m.get("name"))
    # fallback
    return ("Uncategorized", None)

# --------------- Reporting ---------------

def write_csv(rows: List[Dict], out_csv: Path):
    fieldnames = ["date","vendor","amount","category","matcher","notes","source_file","sha1"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

def upsert_sqlite(rows: List[Dict], sqlite_path: Path):
    conn = sqlite3.connect(sqlite_path.as_posix())
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS receipts (
        id INTEGER PRIMARY KEY,
        date TEXT,
        vendor TEXT,
        amount REAL,
        category TEXT,
        matcher TEXT,
        notes TEXT,
        source_file TEXT,
        sha1 TEXT UNIQUE
    )
    """)
    for r in rows:
        cur.execute("""
        INSERT OR IGNORE INTO receipts (date,vendor,amount,category,matcher,notes,source_file,sha1)
        VALUES (?,?,?,?,?,?,?,?)
        """, (r.get("date"), r.get("vendor"), r.get("amount"), r.get("category"),
              r.get("matcher"), r.get("notes"), r.get("source_file"), r.get("sha1")))
    conn.commit()
    conn.close()

def init_duplicates_db(db_path: Path):
    """Initialize the cross-week duplicate detection database"""
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS receipt_fingerprints (
        id INTEGER PRIMARY KEY,
        fingerprint TEXT UNIQUE,
        week_id TEXT,
        date TEXT,
        vendor TEXT,
        amount REAL,
        source_file TEXT,
        first_seen_timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def check_duplicates(db_path: Path, rows: List[Dict], current_week: str) -> List[Dict]:
    """
    Check for duplicates across all weeks based on date, vendor, and amount.
    Returns list of duplicates found (each entry includes the original week and details).
    """
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()

    duplicates = []
    for row in rows:
        fingerprint = compute_receipt_fingerprint(
            row.get("date"),
            row.get("vendor"),
            row.get("amount")
        )

        # Check if this fingerprint exists in a different week
        cur.execute("""
        SELECT week_id, date, vendor, amount, source_file
        FROM receipt_fingerprints
        WHERE fingerprint = ? AND week_id != ?
        """, (fingerprint, current_week))

        result = cur.fetchone()
        if result:
            duplicates.append({
                "current": row,
                "original_week": result[0],
                "original_date": result[1],
                "original_vendor": result[2],
                "original_amount": result[3],
                "original_file": result[4]
            })

    conn.close()
    return duplicates

def register_receipts_in_duplicates_db(db_path: Path, rows: List[Dict], week_id: str):
    """Register all receipts from current week in the duplicates database"""
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()

    timestamp = dt.datetime.now().isoformat()

    for row in rows:
        fingerprint = compute_receipt_fingerprint(
            row.get("date"),
            row.get("vendor"),
            row.get("amount")
        )

        cur.execute("""
        INSERT OR REPLACE INTO receipt_fingerprints
        (fingerprint, week_id, date, vendor, amount, source_file, first_seen_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (fingerprint, week_id, row.get("date"), row.get("vendor"),
              row.get("amount"), row.get("source_file"), timestamp))

    conn.commit()
    conn.close()

def money_fmt(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v is not None else ""

def build_summary_pdf(rows: List[Dict], out_pdf: Path, receipt_page_map: Optional[Dict[str, int]] = None, title="Loss of Use Receipts Summary", use_links: bool = False) -> List[Dict]:
    """
    Build summary PDF with monthly grouping and clickable links to receipts.

    Args:
        rows: List of receipt data
        out_pdf: Output PDF path
        receipt_page_map: Dict mapping source_file to page number in final PDF
        title: Report title
        use_links: Whether to add clickable links (only works in final merged PDF)

    Returns:
        List of link metadata dicts with keys: page, x, y, width, height, target_page
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.colors import blue
    from collections import defaultdict
    import calendar

    receipt_page_map = receipt_page_map or {}
    use_links = use_links and len(receipt_page_map) > 0

    # Track link locations for annotation
    link_metadata = []

    # Group by month and compute totals
    monthly_data = defaultdict(list)
    monthly_totals = defaultdict(float)
    category_totals = defaultdict(float)

    for r in rows:
        date_str = r.get("date") or ""
        cat = r.get("category") or "Uncategorized"
        amt = r.get("amount") or 0.0

        # Extract year-month
        if date_str and len(date_str) >= 7:
            year_month = date_str[:7]  # YYYY-MM
        else:
            year_month = "Unknown"

        monthly_data[year_month].append(r)
        monthly_totals[year_month] += float(amt)
        category_totals[cat] += float(amt)

    c = canvas.Canvas(out_pdf.as_posix(), pagesize=letter)
    width, height = letter
    current_page_num = 0  # Track current page number for link annotations

    # Title page
    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, title)
    y -= 0.3 * inch
    c.setFont("Helvetica", 10)
    timestamp = dt.datetime.now().isoformat(timespec='seconds')
    c.drawString(1 * inch, y, f"Generated: {timestamp}")
    y -= 0.4 * inch

    # Category Totals
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Category Totals")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for cat, amt in sorted(category_totals.items()):
        c.drawString(1.1 * inch, y, f"{cat}: {money_fmt(amt)}")
        y -= 0.2 * inch
        if y < 1.2 * inch:
            c.showPage()
            current_page_num += 1
            y = height - 1 * inch

    # Monthly breakdown
    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Monthly Totals")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)

    for year_month in sorted(monthly_data.keys()):
        if year_month != "Unknown" and len(year_month) == 7:
            year, month = year_month.split("-")
            month_name = calendar.month_name[int(month)]
            display = f"{month_name} {year}"
        else:
            display = year_month

        c.drawString(1.1 * inch, y, f"{display}: {money_fmt(monthly_totals[year_month])}")
        y -= 0.2 * inch
        if y < 1.2 * inch:
            c.showPage()
            current_page_num += 1
            y = height - 1 * inch

    # Line items grouped by month
    for year_month in sorted(monthly_data.keys()):
        month_rows = sorted(monthly_data[year_month], key=lambda x: (x.get("date") or "", x.get("vendor") or ""))

        # Month header
        c.showPage()
        current_page_num += 1
        y = height - 1 * inch
        c.setFont("Helvetica-Bold", 14)

        if year_month != "Unknown" and len(year_month) == 7:
            year, month = year_month.split("-")
            month_name = calendar.month_name[int(month)]
            display = f"{month_name} {year}"
        else:
            display = year_month

        c.drawString(1 * inch, y, display)
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, y - 0.2 * inch, f"Total: {money_fmt(monthly_totals[year_month])}")
        y -= 0.5 * inch

        # Column headers
        c.setFont("Helvetica-Bold", 9)
        c.drawString(1.00 * inch, y, "Date")
        c.drawString(2.10 * inch, y, "Vendor")
        c.drawString(4.30 * inch, y, "Category")
        c.drawRightString(7.50 * inch, y, "Amount")
        y -= 0.15 * inch
        c.line(1.0*inch, y, 7.6*inch, y)
        y -= 0.15 * inch

        # Line items with clickable links
        c.setFont("Helvetica", 9)
        for r in month_rows:
            d = r.get("date") or ""
            v = (r.get("vendor") or "")[:28]
            cat = (r.get("category") or "")[:18]
            amt = money_fmt(r.get("amount"))
            source_file = r.get("source_file") or ""

            # Draw text
            c.drawString(1.00 * inch, y, d)

            # Draw vendor (color blue if it will link in final PDF)
            vendor_x = 2.10 * inch
            if use_links and source_file in receipt_page_map:
                c.setFillColor(blue)
                c.drawString(vendor_x, y, v)
                c.setFillColor("black")

                # Record link metadata for annotation
                # Estimate text width (9pt Helvetica ≈ 0.5 * len(text) points)
                text_width = len(v) * 4.5  # 9pt font, roughly 0.5pt per char
                link_metadata.append({
                    "page": current_page_num,
                    "x": vendor_x,
                    "y": y - 2,  # Slight offset for better clickable area
                    "width": text_width,
                    "height": 11,  # Font height + padding
                    "target_page": receipt_page_map[source_file]
                })
            else:
                c.drawString(vendor_x, y, v)

            c.drawString(4.30 * inch, y, cat)
            c.drawRightString(7.50 * inch, y, amt)
            y -= 0.18 * inch

            if y < 0.8 * inch:
                c.showPage()
                current_page_num += 1
                y = height - 1 * inch
                c.setFont("Helvetica-Bold", 12)
                c.drawString(1 * inch, y, f"{display} (cont.)")
                y -= 0.3 * inch
                c.setFont("Helvetica", 9)

    c.showPage()
    c.save()

    # Return link metadata for annotation
    return link_metadata

def add_pdf_link_annotations(pdf_path: Path, link_metadata: List[Dict]) -> None:
    """
    Add clickable link annotations to a PDF.

    Args:
        pdf_path: Path to PDF file (will be modified in place)
        link_metadata: List of dicts with keys: page, x, y, width, height, target_page
    """
    from pypdf import PdfWriter, PdfReader
    from pypdf.generic import RectangleObject, DictionaryObject, NameObject, NumberObject, ArrayObject

    reader = PdfReader(pdf_path.as_posix())
    writer = PdfWriter()

    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)

    # Add link annotations
    for link in link_metadata:
        page_num = link["page"]
        if page_num >= len(writer.pages):
            continue

        page = writer.pages[page_num]
        page_height = float(page.mediabox.height)

        # Create link annotation
        # PDF coordinates: (0,0) is bottom-left, ReportLab uses same
        x = link["x"]
        y = link["y"]
        width = link["width"]
        height = link["height"]
        target_page = link["target_page"]

        # Create rectangle for clickable area
        rect = RectangleObject([x, y, x + width, y + height])

        # Create link annotation dictionary
        link_dict = DictionaryObject()
        link_dict.update({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Rect"): rect,
            NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),  # No border
            NameObject("/Dest"): ArrayObject([
                writer.pages[target_page].indirect_reference,
                NameObject("/Fit")
            ])
        })

        # Add annotation to page
        if "/Annots" in page:
            page[NameObject("/Annots")].append(writer._add_object(link_dict))
        else:
            page[NameObject("/Annots")] = ArrayObject([writer._add_object(link_dict)])

    # Write back to same file
    with pdf_path.open("wb") as f:
        writer.write(f)

def merge_pdfs(summary_pdf: Path, receipt_files: List[Path], out_pdf: Path, rows: List[Dict]) -> Dict[str, int]:
    """
    Merge summary and receipts into final PDF, tracking page numbers.

    Args:
        summary_pdf: Summary PDF path
        receipt_files: List of receipt PDF paths
        out_pdf: Output merged PDF path
        rows: Receipt data to map source files to PDF files

    Returns:
        Dict mapping source_file to page number in final PDF
    """
    from pypdf import PdfWriter, PdfReader

    writer = PdfWriter()

    # Add summary pages
    summary_reader = PdfReader(summary_pdf.as_posix())
    summary_page_count = len(summary_reader.pages)
    writer.append(summary_reader)

    # Add receipt pages and track page numbers
    receipt_page_map = {}
    current_page = summary_page_count

    # Sort receipt files to match the order in rows (by date, vendor)
    sorted_rows = sorted(rows, key=lambda x: (x.get("date") or "", x.get("vendor") or ""))

    for row in sorted_rows:
        source_file = row.get("source_file", "")
        # Find matching receipt file
        matching_files = [f for f in receipt_files if f.name == source_file or f.stem + ".pdf" == source_file]

        if matching_files:
            receipt_file = matching_files[0]
            try:
                reader = PdfReader(receipt_file.as_posix())
                receipt_page_map[source_file] = current_page
                writer.append(reader)
                current_page += len(reader.pages)
            except Exception as e:
                print(f"[WARN] Skipping {receipt_file.name}: {e}")

    with out_pdf.open("wb") as f:
        writer.write(f)

    return receipt_page_map

# --------------- Main ---------------

@dataclass
class ReceiptEntry:
    date: Optional[str]
    vendor: Optional[str]
    amount: Optional[float]
    category: str
    matcher: Optional[str]
    notes: Optional[str]
    source_file: str
    sha1: str

def process_one_file(path: Path, work_pdf_dir: Path) -> Tuple[str, str, str]:
    """
    Returns (plain_text, search_pdf_path, canonical_ext)
    - For images: OCR directly to text; also convert to single-page searchable PDF via PIL->pytesseract
    - For PDFs: ensure searchable using ocrmypdf (or copy if already has text)
    """
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
            c.showPage(); c.save()
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

def main():
    parser = argparse.ArgumentParser(
        description="Local OCR + categorize + package receipts for insurance submission (weekly batches)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process receipts with all defaults (configure vendor_blacklist in rules.json)
  python process_receipts.py

  # Reprocess an existing subdirectory (processes both incoming + that subdir's processed files)
  python process_receipts.py --subdir 2025-W44

  # Temporarily add to vendor blacklist (in addition to rules.json)
  python process_receipts.py --ignore-vendor "EXTRA NAME"

  # Skip duplicate checking (faster processing)
  python process_receipts.py --skip-duplicate-check

  # Export to SQLite database (in addition to CSV)
  python process_receipts.py --export-db

  # Custom directories
  python process_receipts.py --incoming ./my_receipts --output ./my_output
        """
    )
    parser.add_argument("--incoming", default="./incoming", help="Folder with new receipts (default: ./incoming)")
    parser.add_argument("--output", default="./output", help="Root folder for output batches (default: ./output)")
    parser.add_argument("--subdir", help="Subdirectory identifier (e.g., 2025-W43). Auto-generates current week if not specified")
    parser.add_argument("--rules", default="./rules.json", help="rules.json for category mapping (default: ./rules.json)")
    parser.add_argument("--notes", default="", help="Optional note to include on each entry (e.g., Claim #)")
    parser.add_argument("--ignore-vendor", action="append", help="Ignore this text when extracting vendor (e.g., your name). Can be used multiple times")
    parser.add_argument("--export-db", action="store_true", help="Also export receipts to SQLite database (receipts.sqlite) for this week")
    parser.add_argument("--skip-duplicate-check", action="store_true", help="Skip checking for duplicate receipts across previous weeks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed parsing information for debugging")
    args = parser.parse_args()

    # Lazy import heavy deps only when running
    _lazy_imports()

    # Determine subdirectory identifier (default to current week)
    subdir_id = args.subdir if args.subdir else get_current_week()
    print(f"[INFO] Processing receipts for: {subdir_id}")

    # Setup directory structure
    incoming = Path(args.incoming)
    incoming.mkdir(parents=True, exist_ok=True)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    subdir = output_root / subdir_id
    subdir.mkdir(parents=True, exist_ok=True)

    processed = subdir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    out = subdir / "output"
    out.mkdir(parents=True, exist_ok=True)

    work_pdf_dir = out / "_work_pdfs"
    work_pdf_dir.mkdir(parents=True, exist_ok=True)

    # Initialize duplicate detection database (unless disabled)
    duplicates_db = output_root / "duplicates.db"
    if not args.skip_duplicate_check:
        init_duplicates_db(duplicates_db)

    rules = load_rules(Path(args.rules))

    # Build vendor blacklist from rules.json and command-line arguments
    vendor_blacklist = rules.get("vendor_blacklist", [])

    # Add command-line overrides/additions
    if args.ignore_vendor:
        vendor_blacklist.extend(args.ignore_vendor)

    # Filter out example entries
    vendor_blacklist = [v for v in vendor_blacklist if not v.upper().startswith("EXAMPLE:")]

    if vendor_blacklist:
        print(f"[INFO] Ignoring vendor text: {', '.join(vendor_blacklist)}")

    # Get vendor hints for manual name mapping
    vendor_hints = rules.get("vendor_hints", {})
    if vendor_hints:
        print(f"[INFO] Using {len(vendor_hints)} vendor hint(s)")

    # Initialize LLM cache database
    llm_cache_path = output_root / "llm_cache.db"
    init_llm_cache_db(llm_cache_path)

    # Discover files from both incoming and the week's processed directory
    files_from_incoming = sorted([p for p in incoming.iterdir() if p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS)])

    # Also check processed directory for this week (for reprocessing)
    files_from_processed = []
    if processed.exists():
        for cat_dir in processed.iterdir():
            if cat_dir.is_dir():
                files_from_processed.extend([p for p in cat_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS)])

    files = sorted(files_from_incoming + files_from_processed, key=lambda p: p.name)

    if not files:
        print("No receipt files found in incoming or processed directories.")
        return

    print(f"[INFO] Found {len(files_from_incoming)} file(s) in incoming, {len(files_from_processed)} file(s) in processed")

    rows: List[Dict] = []
    packaged_pdfs: List[Path] = []

    for path in files:
        print(f"[INFO] Processing {path.name}")
        try:
            text, pdf_path, ext = process_one_file(path, work_pdf_dir)
            sha1 = sha1_file(path)

            # Try LLM extraction first (vendor + category in one call)
            # Cache by file hash to avoid re-processing
            categories = rules.get("categories", ["Uncategorized"])
            llm_result = extract_with_llm(text, categories, use_llm=True, file_hash=sha1, cache_path=llm_cache_path)

            vendor = llm_result.get("vendor") or ""
            llm_date = llm_result.get("date")
            category = llm_result.get("category", "Uncategorized")
            llm_confidence = llm_result.get("confidence", 0.0)
            llm_reasoning = llm_result.get("reasoning", "")
            is_cached = llm_result.get("cached", False)
            matcher = f"LLM (confidence: {llm_confidence:.2f}){' [cached]' if is_cached else ''}"

            # Fallback to regex-based extraction if LLM fails or has low confidence
            if not vendor or llm_confidence < 0.3:
                if args.verbose:
                    print(f"  [DEBUG] LLM extraction failed or low confidence, falling back to regex")
                vendor_fallback = parse_vendor(text, blacklist=vendor_blacklist, hints=vendor_hints, rules=rules)
                if vendor_fallback:
                    vendor = vendor_fallback
                    category_fallback, matcher_fallback = categorize(vendor, text, rules)
                    category = category_fallback
                    matcher = matcher_fallback or "Regex fallback"
                # Don't have LLM date in fallback mode
                llm_date = None

            # Always check vendor hints for manual overrides (highest priority)
            text_lower = text.lower()
            for pattern, hint_vendor in vendor_hints.items():
                if pattern.lower() in text_lower:
                    vendor = hint_vendor
                    # Recategorize with hint vendor
                    category, matcher = categorize(vendor, text, rules)
                    matcher = f"{matcher or 'Manual hint'}"
                    break

            # Use LLM-extracted date if available, otherwise fallback to regex
            date = llm_date if llm_date else parse_date(text)
            amount = parse_amount(text)

            # Verbose logging to help debug parsing issues
            if args.verbose:
                print(f"  [DEBUG] Vendor: '{vendor or '(none)'}'")
                print(f"  [DEBUG] Category: {category} (method: {matcher})")
                if is_cached:
                    print(f"  [DEBUG] LLM result: Used cached result (no API call)")
                else:
                    print(f"  [DEBUG] LLM reasoning: {llm_reasoning}")
                date_source = " (from LLM)" if llm_date else " (from regex)"
                print(f"  [DEBUG] Date: {date or '(none)'}{date_source if date else ''}")
                print(f"  [DEBUG] Amount: {f'${amount:.2f}' if amount else '(none)'}")
                if not vendor:
                    print(f"  [DEBUG] First 5 lines of OCR text:")
                    for i, line in enumerate(text.splitlines()[:5], 1):
                        print(f"    {i}: {line[:80]}")
                if not amount:
                    print(f"  [WARN] Could not extract amount. Check OCR quality.")

            row = {
                "date": date,
                "vendor": vendor,
                "amount": amount,
                "category": category,
                "matcher": matcher,
                "notes": args.notes or "",
                "source_file": os.path.basename(pdf_path),
                "sha1": sha1,
            }
            rows.append(row)

            # Move original to processed/<category>/
            cat_dir = processed / slugify(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            dest = cat_dir / path.name

            # Check if file is already in processed (reprocessing scenario)
            if path.parent == cat_dir:
                # Already in the right category directory, no need to move
                print(f"  [INFO] File already in processed/{slugify(category)}/")
            elif path.parent.parent == processed:
                # In processed but wrong category - move to new category
                print(f"  [INFO] Moving from {path.parent.name}/ to {slugify(category)}/")
                if dest.exists() and dest != path:
                    dest = cat_dir / f"{path.stem}_{sha1[:8]}{path.suffix}"
                shutil.move(path.as_posix(), dest.as_posix())
            else:
                # From incoming - move to processed
                if dest.exists():
                    # Avoid overwrite by suffixing sha1
                    dest = cat_dir / f"{path.stem}_{sha1[:8]}{path.suffix}"
                shutil.move(path.as_posix(), dest.as_posix())

            packaged_pdfs.append(Path(pdf_path))

        except Exception as e:
            print(f"[ERROR] Failed {path.name}: {e}")

    # Check for duplicates across all weeks (unless disabled)
    if not args.skip_duplicate_check:
        print(f"[INFO] Checking for duplicates across all weeks...")
        duplicates = check_duplicates(duplicates_db, rows, subdir_id)
        if duplicates:
            print(f"[WARN] Found {len(duplicates)} potential duplicate(s) from previous weeks:")
            for dup in duplicates:
                curr = dup["current"]
                print(f"  - {curr.get('source_file')}: {curr.get('date')} | {curr.get('vendor')} | {money_fmt(curr.get('amount'))}")
                print(f"    Previously in week {dup['original_week']}: {dup['original_file']}")
        else:
            print(f"[OK] No duplicates found")

        # Register current week's receipts in duplicates database
        register_receipts_in_duplicates_db(duplicates_db, rows, subdir_id)
    else:
        print(f"[INFO] Duplicate checking skipped")

    # Write CSV
    out_csv = out / "receipts.csv"
    write_csv(rows, out_csv)
    print(f"[OK] Wrote {out_csv}")

    # Optional SQLite export
    if args.export_db:
        sqlite_path = out / "receipts.sqlite"
        upsert_sqlite(rows, sqlite_path)
        print(f"[OK] Exported to SQLite: {sqlite_path}")

    # Build summary PDF with monthly grouping
    claim_pdf = out / "claim_package.pdf"
    print(f"[INFO] Building claim package with monthly organization...")

    # Create temporary summary to calculate page numbers
    temp_summary = out / "_temp_summary.pdf"
    build_summary_pdf(rows, temp_summary, use_links=False)

    # Merge to get page mapping
    receipt_page_map = merge_pdfs(temp_summary, packaged_pdfs, claim_pdf, rows)

    # Rebuild summary with visual indicators (blue text) for receipts and get link metadata
    summary_pdf = out / "summary.pdf"
    link_metadata = build_summary_pdf(rows, summary_pdf, receipt_page_map=receipt_page_map, use_links=True)
    print(f"[OK] Wrote {summary_pdf} (organized by month)")

    # Rebuild final claim package
    merge_pdfs(summary_pdf, packaged_pdfs, claim_pdf, rows)

    # Add clickable link annotations to the claim package
    if link_metadata:
        print(f"[INFO] Adding {len(link_metadata)} clickable links to claim package...")
        add_pdf_link_annotations(claim_pdf, link_metadata)

    print(f"[OK] Wrote {claim_pdf} (summary + receipts, organized by month)")
    print(f"     → Click blue vendor names to jump to corresponding receipt")
    print(f"     → Receipts appear in same order as line items")

    # Clean up temp file
    if temp_summary.exists():
        temp_summary.unlink()

    print(f"[OK] Processing complete for {subdir_id}. Output in: {out}")

if __name__ == "__main__":
    main()
