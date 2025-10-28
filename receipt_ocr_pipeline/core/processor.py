"""
Main receipt processing orchestration.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from .utils import sha1_file, slugify, IMAGE_EXTS, PDF_EXTS, money_fmt
from .ocr import process_receipt_file
from .parsers import parse_date, parse_amount, parse_vendor
from .llm import extract_with_llm
from .categorization import categorize, load_rules
from .database import (init_llm_cache_db, init_duplicates_db, check_duplicates,
                       register_receipts_in_duplicates_db, remove_week_from_duplicates_db,
                       upsert_sqlite, get_llm_cache_batch, save_llm_cache)
from .reporting import write_csv, build_summary_pdf, merge_pdfs, add_pdf_link_annotations


class ReceiptProcessor:
    """Main processor for receipt OCR and categorization pipeline."""

    def __init__(self, incoming_dir: Path, output_dir: Path,
                 rules_path: Path, subdir_id: str,
                 vendor_blacklist: Optional[List[str]] = None,
                 notes: str = "",
                 export_db: bool = False,
                 verbose: bool = False,
                 llm_provider: str = "openai",
                 llm_model: Optional[str] = None,
                 use_llm: bool = True):
        """
        Initialize receipt processor.

        Args:
            incoming_dir: Directory with new receipts
            output_dir: Root directory for output batches
            rules_path: Path to rules.json
            subdir_id: Subdirectory identifier (e.g., week ID)
            vendor_blacklist: List of strings to exclude from vendor extraction
            notes: Optional notes to add to each entry
            export_db: Whether to export to SQLite
            verbose: Whether to show verbose debugging output
            llm_provider: LLM provider to use ("openai", "anthropic", "azure-openai") - default: openai
            llm_model: LLM model name (uses provider default if not specified)
            use_llm: Whether to use LLM extraction
        """
        self.incoming_dir = incoming_dir
        self.output_dir = output_dir
        self.rules_path = rules_path
        self.subdir_id = subdir_id
        self.vendor_blacklist = vendor_blacklist or []
        self.notes = notes
        self.export_db = export_db
        self.verbose = verbose
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.use_llm = use_llm

        # Setup directories
        self.batch_dir = output_dir / subdir_id  # Main batch directory (e.g., output/2025-W43/)
        self.reports_dir = self.batch_dir / "reports"  # Generated reports (CSV, PDFs)
        self.processed_dir = self.batch_dir / "processed"  # Processed receipts by category
        self.work_pdf_dir = self.batch_dir / "_work_pdfs"  # Temporary PDFs

        # Create directories
        for dir_path in [self.incoming_dir, self.output_dir, self.batch_dir,
                         self.reports_dir, self.processed_dir, self.work_pdf_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load rules
        self.rules = load_rules(rules_path)
        self.vendor_hints = self.rules.get("vendor_hints", {})
        self.categories = self.rules.get("categories", ["Uncategorized"])

        # Initialize databases
        self.duplicates_db = output_dir / "duplicates.db"
        self.llm_cache_db = output_dir / "llm_cache.db"
        init_llm_cache_db(self.llm_cache_db)
        init_duplicates_db(self.duplicates_db)

    def discover_files(self) -> List[Path]:
        """Discover receipt files from incoming and processed directories."""
        # Files from incoming
        files_from_incoming = sorted([
            p for p in self.incoming_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS)
        ])

        # Files from processed (for reprocessing)
        files_from_processed = []
        if self.processed_dir.exists():
            for cat_dir in self.processed_dir.iterdir():
                if cat_dir.is_dir():
                    files_from_processed.extend([
                        p for p in cat_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS)
                    ])

        files = sorted(files_from_incoming + files_from_processed, key=lambda p: p.name)
        print(f"[INFO] Found {len(files_from_incoming)} file(s) in incoming, "
              f"{len(files_from_processed)} file(s) in processed")

        return files

    def process_file(self, path: Path, file_hash: str = None,
                     llm_cache: Dict[str, Dict] = None,
                     new_llm_results: Dict[str, Dict] = None) -> Dict:
        """
        Process a single receipt file.

        Args:
            path: Path to receipt file
            file_hash: Pre-computed SHA1 hash (optional, will compute if not provided)
            llm_cache: Pre-loaded LLM cache dict (optional, for batch optimization)
            new_llm_results: Dict to store new LLM results (optional, for batch saving)

        Returns:
            Dictionary with parsed receipt data
        """
        print(f"[INFO] Processing {path.name}")

        # OCR the file
        text, pdf_path, ext = process_receipt_file(path, self.work_pdf_dir)
        sha1 = file_hash if file_hash else sha1_file(path)

        # Try LLM extraction first
        # Check pre-loaded cache first if available
        if llm_cache is not None and sha1 in llm_cache:
            llm_result = llm_cache[sha1]
        else:
            llm_result = extract_with_llm(
                text, self.categories,
                use_llm=self.use_llm,
                file_hash=sha1,
                cache_path=None,  # Skip DB lookup, we already checked cache
                provider=self.llm_provider,
                model=self.llm_model
            )
            # Store new result for batch saving later
            if new_llm_results is not None and not llm_result.get("cached", False):
                new_llm_results[sha1] = llm_result

        vendor = llm_result.get("vendor") or ""
        llm_date = llm_result.get("date")
        category = llm_result.get("category", "Uncategorized")
        llm_confidence = llm_result.get("confidence", 0.0)
        llm_reasoning = llm_result.get("reasoning", "")
        is_cached = llm_result.get("cached", False)
        matcher = f"LLM (confidence: {llm_confidence:.2f}){' [cached]' if is_cached else ''}"

        # Fallback to regex-based extraction if LLM fails or has low confidence
        if not vendor or llm_confidence < 0.3:
            if self.verbose:
                print(f"  [DEBUG] LLM extraction failed or low confidence, falling back to regex")
            vendor_fallback = parse_vendor(text, blacklist=self.vendor_blacklist,
                                          hints=self.vendor_hints, rules=self.rules)
            if vendor_fallback:
                vendor = vendor_fallback
                category_fallback, matcher_fallback = categorize(vendor, text, self.rules)
                category = category_fallback
                matcher = matcher_fallback or "Regex fallback"
            # Don't have LLM date in fallback mode
            llm_date = None

        # Always check vendor hints for manual overrides (highest priority)
        text_lower = text.lower()
        for pattern, hint_vendor in self.vendor_hints.items():
            if pattern.lower() in text_lower:
                vendor = hint_vendor
                # Recategorize with hint vendor
                category, matcher = categorize(vendor, text, self.rules)
                matcher = f"{matcher or 'Manual hint'}"
                break

        # Use LLM-extracted date if available, otherwise fallback to regex
        date = llm_date if llm_date else parse_date(text)
        amount = parse_amount(text)

        # Verbose logging
        if self.verbose:
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
            "notes": self.notes or "",
            "source_file": os.path.basename(pdf_path),
            "sha1": sha1,
        }

        # Move file to processed directory
        self._move_to_processed(path, category, sha1)

        return row, Path(pdf_path)

    def _move_to_processed(self, path: Path, category: str, sha1: str):
        """Move processed file to appropriate category directory."""
        cat_dir = self.processed_dir / slugify(category)
        cat_dir.mkdir(parents=True, exist_ok=True)
        dest = cat_dir / path.name

        # Check if file is already in processed (reprocessing scenario)
        if path.parent == cat_dir:
            # Already in the right category directory, no need to move
            print(f"  [INFO] File already in processed/{slugify(category)}/")
        elif path.parent.parent == self.processed_dir:
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

    def process_all(self) -> tuple[List[Dict], List[Path]]:
        """
        Process all receipts.

        Returns:
            Tuple of (rows, packaged_pdfs)
        """
        files = self.discover_files()
        if not files:
            print("No receipt files found in incoming or processed directories.")
            return [], []

        # Pre-compute file hashes and pre-load LLM cache in batch (much faster than per-file lookups)
        file_hashes = [sha1_file(f) for f in files]
        llm_cache = get_llm_cache_batch(self.llm_cache_db, file_hashes) if self.use_llm else {}

        rows = []
        packaged_pdfs = []
        new_llm_results = {}  # Track new LLM results to save in batch

        for file_path, file_hash in zip(files, file_hashes):
            try:
                row, pdf_path = self.process_file(file_path, file_hash, llm_cache, new_llm_results)
                rows.append(row)
                packaged_pdfs.append(pdf_path)
            except Exception as e:
                print(f"[ERROR] Failed {file_path.name}: {e}")

        # Save new LLM results to cache in batch
        if new_llm_results:
            for file_hash, result in new_llm_results.items():
                save_llm_cache(
                    self.llm_cache_db, file_hash,
                    result.get("vendor"), result.get("date"),
                    result.get("category"), result.get("confidence", 0.0),
                    result.get("reasoning", "")
                )

        return rows, packaged_pdfs

    def check_duplicates(self, rows: List[Dict]):
        """Check for duplicates and print warnings."""
        # ANSI color codes
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        print(f"[INFO] Checking for duplicates...")
        duplicates = check_duplicates(self.duplicates_db, rows, self.subdir_id)
        if duplicates:
            print(f"{YELLOW}{BOLD}[WARN] Found {len(duplicates)} potential duplicate(s):{RESET}")
            for dup in duplicates:
                curr = dup["current"]
                orig_week = dup['original_week']
                week_label = "same week" if orig_week == self.subdir_id else f"week {orig_week}"
                print(f"{RED}  ⚠ {curr.get('source_file')}: {curr.get('date')} | "
                      f"{curr.get('vendor')} | {money_fmt(curr.get('amount'))}{RESET}")
                print(f"    Previously seen in {week_label}: {dup['original_file']}")
        else:
            print(f"[OK] No duplicates found")

        # Register current week's receipts
        register_receipts_in_duplicates_db(self.duplicates_db, rows, self.subdir_id)

    def generate_reports(self, rows: List[Dict], packaged_pdfs: List[Path]):
        """Generate CSV and PDF reports."""
        # Write CSV
        out_csv = self.reports_dir / "receipts.csv"
        write_csv(rows, out_csv)
        print(f"[OK] Wrote {out_csv}")

        # Optional SQLite export
        if self.export_db:
            sqlite_path = self.reports_dir / "receipts.sqlite"
            upsert_sqlite(rows, sqlite_path)
            print(f"[OK] Exported to SQLite: {sqlite_path}")

        # Build summary PDF
        claim_pdf = self.reports_dir / "claim_package.pdf"
        print(f"[INFO] Building claim package with monthly organization...")

        # Create temporary summary to calculate page numbers
        temp_summary = self.reports_dir / "_temp_summary.pdf"
        build_summary_pdf(rows, temp_summary, use_links=False)

        # Merge to get page mapping
        receipt_page_map = merge_pdfs(temp_summary, packaged_pdfs, claim_pdf, rows)

        # Rebuild summary with visual indicators (blue text) for receipts and get link metadata
        summary_pdf = self.reports_dir / "summary.pdf"
        link_metadata = build_summary_pdf(rows, summary_pdf,
                                         receipt_page_map=receipt_page_map,
                                         use_links=True)
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

        print(f"[OK] Processing complete for {self.subdir_id}. Reports in: {self.reports_dir}")

    def undo(self):
        """
        Undo processing for this week by moving files back to incoming and removing from duplicates DB.
        This is useful if you made a mistake and want to reprocess from scratch without false duplicates.
        """
        # ANSI color codes
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        print(f"{YELLOW}{BOLD}[UNDO] Rolling back processing for {self.subdir_id}...{RESET}")

        # Move files from processed/ back to incoming/
        moved_count = 0
        if self.processed_dir.exists():
            for cat_dir in self.processed_dir.iterdir():
                if cat_dir.is_dir():
                    for file_path in cat_dir.iterdir():
                        if file_path.suffix.lower() in IMAGE_EXTS.union(PDF_EXTS):
                            dest = self.incoming_dir / file_path.name
                            # Avoid overwriting if file already exists in incoming
                            if dest.exists():
                                print(f"[WARN] Skipping {file_path.name} (already exists in incoming)")
                            else:
                                shutil.move(file_path.as_posix(), dest.as_posix())
                                moved_count += 1
                                print(f"[INFO] Moved {file_path.name} back to incoming/")

        # Remove week from duplicates database
        removed_count = remove_week_from_duplicates_db(self.duplicates_db, self.subdir_id)

        # Delete the batch directory
        if self.batch_dir.exists():
            shutil.rmtree(self.batch_dir)
            print(f"[INFO] Removed batch directory: {self.batch_dir}")

        print(f"{YELLOW}{BOLD}[OK] Undo complete:{RESET}")
        print(f"     - Moved {moved_count} file(s) back to incoming/")
        print(f"     - Removed {removed_count} receipt(s) from duplicates database")
        print(f"     - Deleted batch directory")
        print(f"     You can now reprocess this batch from scratch.")
