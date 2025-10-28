#!/usr/bin/env python3
"""
Main CLI entrypoint for receipt OCR pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

from receipt_ocr_pipeline.core.utils import get_current_week
from receipt_ocr_pipeline.core.processor import ReceiptProcessor
from receipt_ocr_pipeline.core.categorization import load_rules


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Local OCR + categorize + package receipts for insurance submission (weekly batches)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process receipts with all defaults
  receipt-ocr

  # Reprocess an existing subdirectory (processes both incoming + that subdir's processed files)
  receipt-ocr --subdir 2025-W44

  # Export to SQLite database (in addition to CSV)
  receipt-ocr --export-db

  # Custom directories
  receipt-ocr --incoming ./my_receipts --output ./my_output
        """
    )
    parser.add_argument("--incoming", default="./incoming",
                       help="Folder with new receipts (default: ./incoming)")
    parser.add_argument("--output", default="./output",
                       help="Root folder for output batches (default: ./output)")
    parser.add_argument("--subdir",
                       help="Subdirectory identifier (e.g., 2025-W43). Auto-generates current week if not specified")
    parser.add_argument("--rules", default="./rules.json",
                       help="rules.json for category mapping (default: ./rules.json)")
    parser.add_argument("--notes", default="",
                       help="Optional note to include on each entry (e.g., Claim #)")
    parser.add_argument("--export-db", action="store_true",
                       help="Also export receipts to SQLite database (receipts.sqlite) for this week")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed parsing information for debugging")

    # LLM configuration
    parser.add_argument("--llm-provider",
                       choices=["openai", "anthropic", "azure-openai"],
                       help="LLM provider to use (default: openai, or LLM_PROVIDER env var)")
    parser.add_argument("--llm-model",
                       help="LLM model to use (uses provider default if not specified, or LLM_MODEL env var)")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM extraction, use only regex-based parsing")

    args = parser.parse_args()

    # Resolve LLM provider from CLI arg or environment variable
    llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", "openai")
    if llm_provider not in ["openai", "anthropic", "azure-openai"]:
        print(f"[ERROR] Invalid LLM provider: {llm_provider}")
        print(f"[ERROR] Must be one of: openai, anthropic, azure-openai")
        return 1

    # Resolve LLM model from CLI arg or environment variable
    llm_model = args.llm_model or os.getenv("LLM_MODEL")

    # Determine subdirectory identifier (default to current week)
    subdir_id = args.subdir if args.subdir else get_current_week()
    print(f"[INFO] Processing receipts for: {subdir_id}")

    # Load rules to get vendor blacklist
    rules = load_rules(Path(args.rules))
    vendor_blacklist = rules.get("vendor_blacklist", [])

    # Filter out example entries
    vendor_blacklist = [v for v in vendor_blacklist if not v.upper().startswith("EXAMPLE:")]

    if vendor_blacklist:
        print(f"[INFO] Ignoring vendor text: {', '.join(vendor_blacklist)}")

    vendor_hints = rules.get("vendor_hints", {})
    if vendor_hints:
        print(f"[INFO] Using {len(vendor_hints)} vendor hint(s)")

    # Show LLM configuration
    if not args.no_llm:
        model_info = llm_model if llm_model else "default"
        env_source = ""
        if not args.llm_provider and os.getenv("LLM_PROVIDER"):
            env_source = " [from LLM_PROVIDER env]"
        elif not args.llm_model and os.getenv("LLM_MODEL"):
            env_source = " [from LLM_MODEL env]"
        print(f"[INFO] LLM: {llm_provider} ({model_info}){env_source}")

    # Create processor
    processor = ReceiptProcessor(
        incoming_dir=Path(args.incoming),
        output_dir=Path(args.output),
        rules_path=Path(args.rules),
        subdir_id=subdir_id,
        vendor_blacklist=vendor_blacklist,
        notes=args.notes,
        export_db=args.export_db,
        verbose=args.verbose,
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_llm=not args.no_llm
    )

    # Process all receipts
    rows, packaged_pdfs = processor.process_all()

    if not rows:
        return 0

    # Check for duplicates
    processor.check_duplicates(rows)

    # Generate reports
    processor.generate_reports(rows, packaged_pdfs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
