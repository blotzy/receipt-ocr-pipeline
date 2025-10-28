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
    with sqlite3.connect(db_path.as_posix()) as conn:
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


def get_llm_cache(db_path: Path, file_hash: str) -> Optional[Dict]:
    """Retrieve cached LLM result from SQLite."""
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT vendor, date, category, confidence, reasoning
                FROM llm_cache
                WHERE file_hash = ?
            """, (file_hash,))
            result = cur.fetchone()

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


def get_llm_cache_batch(db_path: Path, file_hashes: List[str]) -> Dict[str, Dict]:
    """
    Retrieve multiple cached LLM results in a single query.
    More efficient than calling get_llm_cache() multiple times.

    Args:
        db_path: Path to LLM cache database
        file_hashes: List of file hashes to look up

    Returns:
        Dictionary mapping file_hash -> cached result dict
    """
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cur = conn.cursor()

            if not file_hashes:
                return {}

            # Batch query all hashes at once
            placeholders = ','.join('?' * len(file_hashes))
            cur.execute(f"""
                SELECT file_hash, vendor, date, category, confidence, reasoning
                FROM llm_cache
                WHERE file_hash IN ({placeholders})
            """, file_hashes)

            results = {}
            for row in cur.fetchall():
                results[row[0]] = {
                    "vendor": row[1],
                    "date": row[2],
                    "category": row[3],
                    "confidence": row[4],
                    "reasoning": row[5],
                    "cached": True
                }

            return results
    except Exception as e:
        print(f"[WARN] Could not read from LLM cache batch: {e}")
        return {}


def save_llm_cache(db_path: Path, file_hash: str, vendor: Optional[str],
                   date: Optional[str], category: str, confidence: float,
                   reasoning: str):
    """Save LLM extraction result to SQLite cache."""
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO llm_cache
                (file_hash, vendor, date, category, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_hash, vendor, date, category, confidence, reasoning))
            conn.commit()
    except Exception as e:
        print(f"[WARN] Could not save to LLM cache: {e}")


def init_duplicates_db(db_path: Path):
    """Initialize the cross-week duplicate detection database."""
    with sqlite3.connect(db_path.as_posix()) as conn:
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
        # Add index for faster duplicate lookups
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprint
        ON receipt_fingerprints(fingerprint)
        """)
        conn.commit()


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
    with sqlite3.connect(db_path.as_posix()) as conn:
        cur = conn.cursor()

        duplicates = []
        seen_in_current_batch = {}  # Track fingerprints in current processing batch

        # Build list of all fingerprints to check
        fingerprints_to_check = []
        row_by_fingerprint = {}

        for row in rows:
            fingerprint = compute_receipt_fingerprint(
                row.get("date"),
                row.get("vendor"),
                row.get("amount")
            )
            fingerprints_to_check.append(fingerprint)
            row_by_fingerprint[fingerprint] = row

        # Batch query all fingerprints at once (more efficient than N queries)
        if fingerprints_to_check:
            placeholders = ','.join('?' * len(fingerprints_to_check))
            cur.execute(f"""
            SELECT fingerprint, week_id, date, vendor, amount, source_file, file_hash
            FROM receipt_fingerprints
            WHERE fingerprint IN ({placeholders})
            """, fingerprints_to_check)

            db_results = {result[0]: result[1:] for result in cur.fetchall()}
        else:
            db_results = {}

        # Now check each row
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
            if fingerprint in db_results:
                result = db_results[fingerprint]
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

    return duplicates


def register_receipts_in_duplicates_db(db_path: Path, rows: List[Dict], week_id: str):
    """Register all receipts from current week in the duplicates database."""
    with sqlite3.connect(db_path.as_posix()) as conn:
        cur = conn.cursor()
        timestamp = dt.datetime.now().isoformat()

        # Use executemany for batch insert (more efficient)
        data = []
        for row in rows:
            fingerprint = compute_receipt_fingerprint(
                row.get("date"),
                row.get("vendor"),
                row.get("amount")
            )
            data.append((
                fingerprint, week_id, row.get("date"), row.get("vendor"),
                row.get("amount"), row.get("source_file"), row.get("sha1"), timestamp
            ))

        cur.executemany("""
        INSERT OR REPLACE INTO receipt_fingerprints
        (fingerprint, week_id, date, vendor, amount, source_file, file_hash, first_seen_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        conn.commit()


def remove_week_from_duplicates_db(db_path: Path, week_id: str) -> int:
    """
    Remove all receipts for a specific week from the duplicates database.

    Args:
        db_path: Path to duplicates database
        week_id: Week identifier to remove (e.g., "2025-W43")

    Returns:
        Number of receipts removed
    """
    with sqlite3.connect(db_path.as_posix()) as conn:
        cur = conn.cursor()

        # Count how many will be removed
        cur.execute("SELECT COUNT(*) FROM receipt_fingerprints WHERE week_id = ?", (week_id,))
        count = cur.fetchone()[0]

        # Remove them
        cur.execute("DELETE FROM receipt_fingerprints WHERE week_id = ?", (week_id,))

        conn.commit()

    return count


def upsert_sqlite(rows: List[Dict], sqlite_path: Path):
    """Save receipts to SQLite database."""
    with sqlite3.connect(sqlite_path.as_posix()) as conn:
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

        # Use executemany for batch insert (more efficient)
        data = [(r.get("date"), r.get("vendor"), r.get("amount"), r.get("category"),
                 r.get("matcher"), r.get("notes"), r.get("source_file"), r.get("sha1"))
                for r in rows]

        cur.executemany("""
        INSERT OR IGNORE INTO receipts (date,vendor,amount,category,matcher,notes,source_file,sha1)
        VALUES (?,?,?,?,?,?,?,?)
        """, data)

        conn.commit()
