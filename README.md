# Receipt OCR Pipeline

A local, scriptable pipeline to process receipt scans into structured data and submission-ready PDF packages.

## Features

- **OCR Processing**: Automatically extract text from receipt images and PDFs using Tesseract
- **LLM-Powered Extraction**: Use Claude AI to intelligently extract vendor names, dates, and categories
- **Smart Categorization**: Rule-based categorization with regex patterns and manual hints
- **Duplicate Detection**: Cross-week duplicate detection based on date, vendor, and amount
- **Professional Reports**: Generate CSV exports and linked PDF packages organized by month
- **Caching**: LLM results are cached to minimize API costs

## Package Structure

```
receipt-ocr-pipeline/
├── receipt_ocr_pipeline/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py              # CLI entrypoint
│   └── core/
│       ├── __init__.py
│       ├── categorization.py    # Rule-based categorization
│       ├── database.py          # SQLite operations (caching, duplicates)
│       ├── llm.py               # Claude AI extraction
│       ├── models.py            # Data models
│       ├── ocr.py               # OCR functionality
│       ├── parsers.py           # Text parsing (dates, amounts, vendors)
│       ├── processor.py         # Main orchestration
│       ├── reporting.py         # PDF and CSV generation
│       └── utils.py             # Utilities and constants
├── process_receipts.py          # Original script (deprecated)
├── pyproject.toml               # Package configuration
├── setup.py                     # Setup script
└── README.md                    # This file
```

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Tesseract OCR** (binary):
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **ocrmypdf** (recommended for better PDF OCR):
   ```bash
   pip install ocrmypdf
   ```

### Install Package

```bash
# Install in development mode (recommended)
pip install -e .

# Or install from source
pip install .

# Install with development dependencies
pip install -e ".[dev]"
```

### Set up API Keys

The package supports multiple LLM providers. Set the appropriate API key for your chosen provider:

**OpenAI (default):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Anthropic Claude:**
```bash
# Install Anthropic support
pip install -e ".[anthropic]"

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Azure OpenAI:**
```bash
# Set Azure configuration
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

## Usage

### Basic Usage

```bash
# Process receipts with default settings
receipt-ocr

# Process with custom directories
receipt-ocr --incoming ./my_receipts --output ./my_output

# Specify a custom week/batch identifier
receipt-ocr --subdir 2025-W44

# Undo/rollback a week's processing (moves files back to incoming, removes from duplicates DB)
receipt-ocr --subdir 2025-W44 --undo
```

### Advanced Options

```bash
# Export to SQLite database
receipt-ocr --export-db

# Verbose debugging output
receipt-ocr -v

# Custom rules file
receipt-ocr --rules ./custom_rules.json
```

### Undo Processing

If you made a mistake while processing a week and want to start over without getting false duplicate warnings:

```bash
# Undo processing for a specific week
receipt-ocr --subdir 2025-W43 --undo

# This will:
# - Move all receipts from output/2025-W43/processed/ back to incoming/
# - Remove all 2025-W43 entries from the duplicates database
# - Delete the output/2025-W43/ directory
# - Allow you to reprocess from scratch without duplicate warnings
```

### LLM Provider Options

```bash
# Use default (OpenAI)
receipt-ocr

# Use Anthropic Claude instead
receipt-ocr --llm-provider anthropic

# Use a specific model
receipt-ocr --llm-provider openai --llm-model gpt-4o

# Use Azure OpenAI
receipt-ocr --llm-provider azure-openai --llm-model gpt-4o-mini

# Disable LLM extraction entirely (use only regex-based parsing)
receipt-ocr --no-llm
```

**Environment Variable Configuration:**

You can also set the LLM provider and model via environment variables. CLI arguments will override these:

```bash
# Set provider via environment variable
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-sonnet-20241022
receipt-ocr  # Uses Anthropic Sonnet

# CLI argument overrides environment variable
export LLM_PROVIDER=anthropic
receipt-ocr --llm-provider openai  # Uses OpenAI instead

# Combine with API keys for complete configuration
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-haiku-20241022
export ANTHROPIC_API_KEY=your-key-here
receipt-ocr  # Ready to go!
```

**Supported Providers:**
- `openai` (default) - Uses GPT-4o-mini by default
- `anthropic` - Uses Claude 3.5 Haiku by default
- `azure-openai` - Uses GPT-4o-mini by default

**Popular Models:**
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`
- Anthropic: `claude-3-5-haiku-20241022`, `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- Azure OpenAI: Deployment names from your Azure resource

### Using as a Library

```python
from pathlib import Path
from receipt_ocr_pipeline.core.processor import ReceiptProcessor
from receipt_ocr_pipeline.core.utils import get_current_week

# Create processor with default settings (OpenAI)
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id=get_current_week(),
    verbose=True
)

# Or use Anthropic Claude instead
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id=get_current_week(),
    llm_provider="anthropic",
    llm_model="claude-3-5-haiku-20241022",
    verbose=True
)

# Process all receipts
rows, packaged_pdfs = processor.process_all()

# Check for duplicates
processor.check_duplicates(rows)

# Generate reports
processor.generate_reports(rows, packaged_pdfs)
```

## Configuration

### rules.json

Configure categories and matching rules:
```json
{
  "categories": [
    "Accommodations",
    "Food/Grocery",
    "Transportation",
    "Entertainment",
    "Other"
  ],
  "vendor_blacklist": [
    "YOUR NAME HERE",
    "CUSTOMER NAME"
  ],
  "vendor_hints": {
    "airbnb": "Airbnb",
    "inn at thorn": "Inn at Thorn Hill"
  },
  "matchers": [
    {
      "name": "Hotels",
      "any": [
        {"vendor_re": "HILTON|MARRIOTT|HYATT|HOTEL|INN|RESORT"}
      ],
      "category": "Accommodations"
    },
    {
      "name": "Restaurants",
      "any": [
        {"text_re": "RESTAURANT|CAFE|DINER"}
      ],
      "category": "Food/Grocery"
    }
  ]
}
```

## Workflow

1. **Place receipts** in the `incoming/` directory (images or PDFs)
2. **Run the pipeline**: `receipt-ocr`
3. **Review outputs** in `output/<week-id>/reports/`:
   - `receipts.csv` - Structured data
   - `summary.pdf` - Category totals and line items organized by month
   - `claim_package.pdf` - Summary + all receipts with clickable links
   - `receipts.sqlite` - SQLite database (if `--export-db` is used)
4. **Processed files** are moved to `output/<week-id>/processed/<category>/`

## Output Structure

```
output/
  2025-W43/                      # Batch directory (one per week/run)
    reports/                     # Generated reports
      receipts.csv               # Structured data export
      summary.pdf                # Category/monthly summary with line items
      claim_package.pdf          # Complete package (summary + all receipts)
      receipts.sqlite            # SQLite export (optional, with --export-db)
    processed/                   # Categorized original receipts
      accommodations/
        hotel-receipt.pdf
      food-grocery/
        restaurant-receipt.jpg
      transportation/
        uber-receipt.pdf
    _work_pdfs/                  # Temporary PDFs (for processing)
  2025-W44/                      # Next week's batch
    reports/
      ...
    processed/
      ...
  duplicates.db                  # Cross-week duplicate detection database
  llm_cache.db                   # LLM result cache (saves API costs)
```

## Output Files

### receipts.csv
Structured data with columns:
- date, vendor, amount, category, matcher, notes, source_file, sha1

### summary.pdf
- Category totals
- Monthly totals
- Detailed line items grouped by month
- Blue vendor names indicate clickable links to receipts

### claim_package.pdf
- Complete package with summary followed by all receipts
- Click blue vendor names to jump to corresponding receipt
- Receipts appear in chronological order

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

```bash
# Format code
black receipt_ocr_pipeline/

# Lint code
ruff check receipt_ocr_pipeline/
```

## Module Overview

### Core Modules

- **categorization.py**: Rule-based receipt categorization using regex patterns
- **database.py**: SQLite operations for caching LLM results and detecting duplicates
- **llm.py**: Claude AI integration for intelligent vendor/category extraction
- **models.py**: Data models (ReceiptEntry dataclass)
- **ocr.py**: OCR processing for images and PDFs using Tesseract
- **parsers.py**: Text parsing for dates, amounts, and vendor names using regex
- **processor.py**: Main orchestration class that coordinates the entire pipeline
- **reporting.py**: PDF and CSV generation with clickable links
- **utils.py**: Shared utilities and constants

### CLI Module

- **main.py**: Command-line interface entrypoint

## Migration from Original Script

The original `process_receipts.py` is still available but deprecated. To migrate:

1. Install the package: `pip install -e .`
2. Replace `python process_receipts.py` with `receipt-ocr`
3. All functionality is preserved with the same command-line arguments

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
