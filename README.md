# Local Receipt OCR + Categorize + Claim PDF (Weekly Batches)

This is a local, scriptable pipeline to OCR your receipt scans, extract vendor/date/amount using **AI-powered extraction (LLM)** with regex fallback, categorize them, and generate weekly reports with duplicate detection across all weeks:
- `weeks/<week_id>/output/receipts.csv` (structured data for that week)
- `weeks/<week_id>/output/summary.pdf` (category totals + line items)
- `weeks/<week_id>/output/claim_package.pdf` (summary followed by all receipt PDFs)
- `weeks/duplicates.db` (cross-week duplicate detection)
- Optional per-week `receipts.sqlite`

## 1) Install prerequisites

### macOS (Homebrew)
```bash
brew install tesseract ocrmypdf ghostscript
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API key for LLM extraction (optional but recommended)
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Debian/Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr ocrmypdf ghostscript python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Folder layout (Batched Processing)
```
receipt_processor/
  incoming/              # drop images or PDFs here
  output/                # root for all batches
    2025-W43/            # batch subdirectory (defaults to current week)
      processed/         # receipts moved here, grouped by category
      output/            # CSV & PDFs for this batch
    2025-W44/
      processed/
      output/
    duplicates.db        # tracks duplicates across ALL batches
    llm_cache.db         # LLM extraction cache (shared across all batches)
  rules.json             # your categorization rules
  process_receipts.py
```

Each batch is self-contained. You can reprocess any batch without affecting others.

## 3) AI-Powered Extraction

The script uses **Claude (LLM)** to intelligently extract vendor names and categorize receipts:

### How it works:
1. **LLM extraction first** (Claude Haiku for speed and cost efficiency)
   - Analyzes OCR text to identify vendor and category
   - Handles complex cases (e.g., "Inn at Thorn Hill", "Airbnb")
   - Provides confidence score and reasoning

2. **Fallback to regex** if LLM fails or has low confidence (<0.3)
   - Uses traditional pattern matching from `rules.json`
   - Ensures processing never fails

3. **Manual hints override** (highest priority)
   - Check `vendor_hints` in `rules.json` for explicit mappings

### Setup:
```bash
# Set your API key (get one from https://console.anthropic.com)
export ANTHROPIC_API_KEY="your-api-key-here"

# Or add to your shell profile (~/.zshrc or ~/.bashrc)
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.zshrc
```

**Cost**: ~$0.001-0.01 per receipt (Claude Haiku is very cost-effective)

**Caching**: LLM results are cached in `output/llm_cache.db` (SQLite) by file hash
- Reprocessing the same receipt = **no API call, no cost**
- Only pays for LLM when processing new/changed receipts
- Cache is shared across all batches
- Queryable: `sqlite3 output/llm_cache.db "SELECT * FROM llm_cache WHERE category='Grocery'"`

**Without API key**: Script falls back to regex-based extraction automatically

## 4) Configure rules
Edit `rules.json` to configure:
- **Categories**: Your expense categories (Accommodations, Food, Grocery, etc.)
- **Vendor blacklist**: Your name(s) to exclude from vendor extraction
- **Vendor hints**: Manual overrides for specific vendors (highest priority)
- **Matchers**: Fallback regex rules for categorization

Example `rules.json`:
```json
{
  "categories": ["Accommodations", "Food", "Grocery", "Uncategorized"],
  "vendor_blacklist": ["JOHN SMITH", "J SMITH"],
  "vendor_hints": {
    "safeway": "Safeway",
    "whole foods": "Whole Foods"
  },
  "matchers": [
    {
      "name": "Hotels - Boutique & Independent",
      "vendor_extract": [
        {"pattern": "([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+Inn"},
        {"pattern": "([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+Hotel"}
      ],
      "any": [
        {"vendor_re": "\\bHOTEL\\b|\\bINN\\b|\\bMOTEL\\b"},
        {"text_re": "\\bHOTEL\\b|\\bINN\\b|\\bMOTEL\\b"}
      ],
      "category": "Accommodations"
    }
  ]
}
```

**Important**:
- **`vendor_blacklist`**: Your name(s) to prevent from being extracted as vendor
- **`vendor_hints`**: Manual text → vendor name mapping (checked first!)
- **`vendor_extract`** (in matchers): Regex patterns to extract business names from OCR text
  - Example: `"([A-Z][a-z]+)\\s+Inn"` extracts "Quality Inn", "Hampton Inn", etc.
- **Categorization rules**:
  - `vendor_re`: Matches against the extracted vendor name
  - `text_re`: Matches against the full OCR text (more reliable!)
  - Use `text_re` for better matching when vendor extraction is unreliable

## 5) Run (Weekly Processing)

### Simple Usage (All Defaults)
```bash
# Just run it! Uses ./incoming, ./weeks, and ./rules.json
python process_receipts.py
```

This will:
- Look for receipts in `./incoming/`
- Store weekly batches in `./weeks/<current-week>/`
- Use categorization rules from `./rules.json`
- Automatically check for duplicates across all weeks

### Advanced Options
```bash
# Debug parsing issues (shows vendor, date, amount for each receipt)
python process_receipts.py --verbose

# Temporarily ignore additional vendor text (adds to rules.json blacklist)
python process_receipts.py --ignore-vendor "TEMPORARY NAME"

# Specify a particular subdirectory/batch
python process_receipts.py --subdir 2025-W44

# Add notes to all receipts
python process_receipts.py --notes "Claim #12345 (Loss of Use)"

# Skip duplicate checking (faster processing)
python process_receipts.py --skip-duplicate-check

# Export to SQLite database (in addition to CSV)
python process_receipts.py --export-db

# Custom directories
python process_receipts.py --incoming ./my_receipts --output ./my_output --rules ./my_rules.json
```

**Duplicate Detection** (enabled by default): The script automatically checks all previous batches and warns if the same receipt (same date, vendor, and amount) appears in multiple batches. Use `--skip-duplicate-check` to skip this check.

## 6) What you get (Per Batch)
For each batch (e.g., `output/2025-W43/`):
- `output/receipts.csv` with columns: date, vendor, amount, category, matcher, notes, source_file, sha1
- `output/summary.pdf` with category totals and line items organized by month
- `output/claim_package.pdf` merges the summary + every processed receipt PDF in order (clickable blue vendor names!)
- `processed/<category>/...` holds your original files, organized by category

Global files:
- `output/duplicates.db` tracks all receipts across all batches to detect duplicates
- `output/llm_cache.db` caches LLM extraction results for cost savings

## Tips
- **Simplest workflow**: Just drop receipts in `incoming/` and run `python process_receipts.py` - that's it!
- **Batch workflow**: Collect receipts for a period, drop them in `incoming/`, run the script. Each batch gets its own independent report (defaults to current week).
- **Debugging parsing**: Use `--verbose` to see what vendor, date, and amount were extracted from each receipt. Shows LLM reasoning and cache hits.
- **Vendor extraction issues**:
  - With LLM: Usually handles complex cases automatically
  - Add your name to `vendor_blacklist` in `rules.json` to prevent it from being extracted
  - Use `vendor_hints` to manually map text patterns to vendor names (e.g., `"safeway": "Safeway"`)
  - LLM extraction priority: vendor hints → LLM → regex fallback
- **Date extraction**: LLM automatically identifies transaction dates vs. print dates (especially useful for online orders)
- **Reprocessing a batch**: Just run the script again with the same `--subdir`. It automatically reprocesses all files in that batch's `processed/` folder along with any new files in `incoming/`. Perfect for:
  - Fixing vendor extraction by updating `vendor_blacklist` or rules
  - Adjusting categorization rules
  - Adding forgotten receipts to an existing batch
  - Regenerating reports after any changes
  - **Cost**: Uses cached LLM results (no API calls on reprocessing!)
- **Duplicate detection**: Enabled by default - detects receipts with the same date, vendor, and amount across different weeks. Use `--skip-duplicate-check` to skip.
- **SQLite export**: Use `--export-db` to create a SQLite database for the week (in addition to CSV). Useful for SQL queries or integrations.
- **Improve OCR**: Scan at 300–400 DPI; use `ocrmypdf` to add text layers to PDFs.
- **Extend rules**: Add your frequent vendors to `rules.json` (e.g., specific hotels, restaurants, local grocery stores).
- **Re-running**: You can reprocess any week without affecting others. The duplicate database tracks everything globally.
- **Help**: Run `python process_receipts.py --help` to see all options and examples.
