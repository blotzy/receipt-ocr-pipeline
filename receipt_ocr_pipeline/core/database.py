"""
Database operations for receipt storage and caching.
"""

import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional

from .utils import compute_receipt_fingerprint


def init_llm_cache_db(db_path: Path):
    """Initialize SQLite database for LLM extraction cache."""
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
    """Retrieve cached LLM result from SQLite."""
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


def save_llm_cache(db_path: Path, file_hash: str, vendor: Optional[str],
                   date: Optional[str], category: str, confidence: float,
                   reasoning: str):
    """Save LLM extraction result to SQLite cache."""
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


def init_duplicates_db(db_path: Path):
    """Initialize the cross-week duplicate detection database."""
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS receipt_fingerprints (
        id INTEGER PRIMARY KEY,
        fingerprint TEXT,
        week_id TEXT,
        date TEXT,
        vendor TEXT,
        amount REAL,
        source_file TEXT,
        file_hash TEXT,
        first_seen_timestamp TEXT,
        UNIQUE(fingerprint, file_hash)
    )
    """)
    conn.commit()
    conn.close()


def check_duplicates(db_path: Path, rows: List[Dict], current_week: str) -> List[Dict]:
    """
    Check for duplicates based on date, vendor, and amount.
    Detects duplicates both within the same week and across different weeks.
    Uses file_hash to allow idempotent reprocessing of the same file.

    Args:
        db_path: Path to duplicates database
        rows: List of receipt data dictionaries
        current_week: Current week identifier

    Returns:
        List of duplicates found (each entry includes the original week and details).
    """
    conn = sqlite3.connect(db_path.as_posix())
    cur = conn.cursor()

    duplicates = []
    seen_in_current_batch = {}  # Track fingerprints in current processing batch

    for row in rows:
        fingerprint = compute_receipt_fingerprint(
            row.get("date"),
            row.get("vendor"),
            row.get("amount")
        )
        current_file_hash = row.get("sha1")

        # Check within current batch first (for duplicates in same processing run)
        if fingerprint in seen_in_current_batch:
            prev_file_hash, prev_file = seen_in_current_batch[fingerprint]
            # Only warn if different files (different hash)
            if prev_file_hash != current_file_hash:
                duplicates.append({
                    "current": row,
                    "original_week": current_week,
                    "original_date": row.get("date"),
                    "original_vendor": row.get("vendor"),
                    "original_amount": row.get("amount"),
                    "original_file": prev_file
                })

        # Check against previously registered receipts in database
        cur.execute("""
        SELECT week_id, date, vendor, amount, source_file, file_hash
        FROM receipt_fingerprints
        WHERE fingerprint = ?
        """, (fingerprint,))

        result = cur.fetchone()
        if result:
            db_file_hash = result[5]
            # Only warn if it's a different file (different hash)
            if db_file_hash != current_file_hash:
                duplicates.append({
                    "current": row,
                    "original_week": result[0],
                    "original_date": result[1],
                    "original_vendor": result[2],
                    "original_amount": result[3],
                    "original_file": result[4]
                })

        # Track this fingerprint in current batch
        seen_in_current_batch[fingerprint] = (current_file_hash, row.get("source_file"))

    conn.close()
    return duplicates


def register_receipts_in_duplicates_db(db_path: Path, rows: List[Dict], week_id: str):
    """Register all receipts from current week in the duplicates database."""
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
        (fingerprint, week_id, date, vendor, amount, source_file, file_hash, first_seen_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (fingerprint, week_id, row.get("date"), row.get("vendor"),
              row.get("amount"), row.get("source_file"), row.get("sha1"), timestamp))

    conn.commit()
    conn.close()


def upsert_sqlite(rows: List[Dict], sqlite_path: Path):
    """Save receipts to SQLite database."""
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
