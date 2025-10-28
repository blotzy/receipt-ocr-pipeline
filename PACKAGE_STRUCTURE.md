# Package Structure Overview

This document describes the refactored package structure for the receipt OCR pipeline.

## New Package Structure

```
receipt_ocr_pipeline/
├── __init__.py                 # Package root
├── cli/
│   ├── __init__.py
│   └── main.py                 # CLI entrypoint (107 lines)
└── core/
    ├── __init__.py
    ├── categorization.py       # Rule-based categorization (75 lines)
    ├── database.py             # SQLite operations (159 lines)
    ├── llm.py                  # Claude AI extraction (149 lines)
    ├── models.py               # Data models (20 lines)
    ├── ocr.py                  # OCR functionality (128 lines)
    ├── parsers.py              # Text parsing (172 lines)
    ├── processor.py            # Main orchestration (253 lines)
    ├── reporting.py            # PDF/CSV generation (318 lines)
    └── utils.py                # Utilities and constants (93 lines)
```

## Module Responsibilities

### Core Modules

1. **utils.py**
   - Constants (IMAGE_EXTS, PDF_EXTS, DATE_PATTERNS, AMOUNT_PATTERNS)
   - Utility functions (slugify, normalize_amount, sha1_file, money_fmt, etc.)
   - Week management

2. **models.py**
   - ReceiptEntry dataclass
   - Data structure definitions

3. **parsers.py**
   - parse_date() - Extract dates from text
   - parse_amount() - Extract amounts from text
   - parse_vendor() - Extract vendor names with heuristics

4. **ocr.py**
   - ensure_ocr_for_pdf() - Make PDFs searchable
   - ocr_image_to_text() - OCR images to text
   - pdf_to_text() - Extract text from PDFs
   - process_receipt_file() - Main file processing

5. **llm.py**
   - extract_with_llm() - Claude AI extraction
   - LLM client management (lazy loading)
   - Handles caching integration

6. **categorization.py**
   - load_rules() - Load rules.json
   - categorize() - Rule-based categorization

7. **database.py**
   - LLM cache operations (init, get, save)
   - Duplicate detection database
   - Receipt storage (SQLite export)

8. **reporting.py**
   - write_csv() - CSV export
   - build_summary_pdf() - Generate summary PDFs
   - merge_pdfs() - Merge summary + receipts
   - add_pdf_link_annotations() - Add clickable links

9. **processor.py**
   - ReceiptProcessor class - Main orchestration
   - Coordinates all modules
   - Handles workflow (discover → process → check duplicates → report)

### CLI Module

1. **main.py**
   - Command-line argument parsing
   - Entry point for `receipt-ocr` command
   - Calls ReceiptProcessor

## Installation

The package can be installed using pip:

```bash
# Development mode (recommended)
pip install -e .

# Or regular install
pip install .
```

This creates a `receipt-ocr` command that can be run from anywhere.

## Usage

### As a Command-Line Tool

```bash
receipt-ocr --incoming ./receipts --output ./processed
```

### As a Library

```python
from receipt_ocr_pipeline.core.processor import ReceiptProcessor
from pathlib import Path

processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id="2025-W43"
)

rows, pdfs = processor.process_all()
processor.check_duplicates(rows)
processor.generate_reports(rows, pdfs)
```

## Benefits

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Modules can be tested independently
3. **Reusability**: Can import and use specific functions
4. **Extensibility**: Easy to add new features or modify existing ones
5. **Professional**: Follows Python packaging best practices

## Migration

The original `process_receipts.py` is still available for backward compatibility, but users should migrate to:

```bash
receipt-ocr [options]
```

All command-line arguments remain the same.

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Formatting

```bash
black receipt_ocr_pipeline/
ruff check receipt_ocr_pipeline/
```

## Dependencies

Defined in `pyproject.toml`:
- pytesseract
- Pillow
- PyMuPDF
- reportlab
- pypdf
- anthropic

## License

MIT License
