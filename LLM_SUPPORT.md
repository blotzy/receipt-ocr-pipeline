# LLM Provider Support

The receipt OCR pipeline now supports multiple LLM providers for intelligent extraction of vendor names, dates, and categories from receipts.

## Supported Providers

### 1. OpenAI (Default)

**Default Model:** `gpt-4o-mini`

**Setup:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Usage:**
```bash
# Uses OpenAI by default
receipt-ocr

# Explicitly specify OpenAI
receipt-ocr --llm-provider openai

# Use a different OpenAI model
receipt-ocr --llm-provider openai --llm-model gpt-4o
```

**Available Models:**
- `gpt-4o-mini` (fast, cost-effective - default)
- `gpt-4o` (higher quality)
- `gpt-4-turbo` (legacy, still supported)

**Cost:** ~$0.0001-0.002 per receipt with gpt-4o-mini

### 2. Anthropic Claude

**Default Model:** `claude-3-5-haiku-20241022`

**Setup:**
```bash
# Install Anthropic support
pip install -e ".[anthropic]"

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Usage:**
```bash
# Use Anthropic
receipt-ocr --llm-provider anthropic

# Use a specific model
receipt-ocr --llm-provider anthropic --llm-model claude-3-5-sonnet-20241022
```

**Available Models:**
- `claude-3-5-haiku-20241022` (fast, cost-effective - default)
- `claude-3-5-sonnet-20241022` (balanced performance)
- `claude-3-opus-20240229` (highest quality)

**Cost:** ~$0.001-0.01 per receipt with Haiku

### 3. Azure OpenAI

**Default Model:** `gpt-4o-mini`

**Setup:**
```bash
# Set Azure configuration
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

**Usage:**
```bash
# Use Azure OpenAI
receipt-ocr --llm-provider azure-openai

# Use a specific deployment
receipt-ocr --llm-provider azure-openai --llm-model your-deployment-name
```

**Note:** The `--llm-model` should match your Azure deployment name, not the underlying model.

## Environment Variable Configuration

You can configure the LLM provider and model using environment variables, with CLI arguments taking precedence:

```bash
# Set provider and model via environment variables
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_API_KEY=your-key-here

# Now just run without arguments
receipt-ocr  # Uses Anthropic Sonnet

# CLI arguments override environment variables
receipt-ocr --llm-provider openai  # Uses OpenAI instead
```

**Priority Order:**
1. CLI arguments (`--llm-provider`, `--llm-model`) - highest priority
2. Environment variables (`LLM_PROVIDER`, `LLM_MODEL`)
3. Built-in defaults (OpenAI, gpt-4o-mini)

**Example .envrc or .env file:**
```bash
# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# LLM Configuration
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
```

This is especially useful for:
- **Consistent configuration** across multiple runs
- **Team environments** where everyone uses the same provider
- **CI/CD pipelines** with centralized configuration
- **Docker containers** with environment-based config

## Programmatic Usage

### Python API

```python
from pathlib import Path
from receipt_ocr_pipeline.core.processor import ReceiptProcessor

# OpenAI (default)
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id="2025-W43"
)

# Anthropic
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id="2025-W43",
    llm_provider="anthropic",
    llm_model="claude-3-5-haiku-20241022"
)

# Azure OpenAI
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id="2025-W43",
    llm_provider="azure-openai",
    llm_model="your-deployment-name"
)

# Disable LLM (regex-based only)
processor = ReceiptProcessor(
    incoming_dir=Path("./incoming"),
    output_dir=Path("./output"),
    rules_path=Path("./rules.json"),
    subdir_id="2025-W43",
    use_llm=False
)
```

### Direct LLM Function

```python
from receipt_ocr_pipeline.core.llm import extract_with_llm

# OpenAI (default)
result = extract_with_llm(
    ocr_text="receipt text here...",
    categories=["Food", "Gas", "Other"],
    provider="openai",
    model="gpt-4o-mini"
)

# Anthropic
result = extract_with_llm(
    ocr_text="receipt text here...",
    categories=["Food", "Gas", "Other"],
    provider="anthropic",
    model="claude-3-5-haiku-20241022"
)

# Returns:
# {
#     "vendor": "Safeway",
#     "date": "2025-10-27",
#     "category": "Food",
#     "confidence": 0.95,
#     "reasoning": "Store name clearly visible, dated receipt",
#     "cached": False
# }
```

## Caching

All LLM results are cached in `output/llm_cache.db` based on file hash. This means:

- **Same file reprocessed = No API call, no cost**
- Cache is shared across all providers
- Cache persists between runs
- Only new or modified receipts trigger LLM calls

To view cache:
```bash
sqlite3 output/llm_cache.db "SELECT * FROM llm_cache LIMIT 10"
```

## Fallback Behavior

The pipeline uses a multi-tier extraction strategy:

1. **LLM First** - Tries configured LLM provider
2. **Confidence Check** - If confidence < 0.3, falls back to regex
3. **Regex Fallback** - Uses traditional pattern matching from rules.json
4. **Vendor Hints** - Manual overrides always take precedence

This ensures extraction never fails, even if:
- API is down
- API key is missing
- LLM returns low-confidence results

## Cost Optimization

### Tips for Minimizing Costs

1. **Use Fast Models**
   - OpenAI: `gpt-4o-mini` (default)
   - Anthropic: `claude-3-5-haiku-20241022`

2. **Enable Caching** (enabled by default)
   - Reprocessing = free
   - Only new receipts cost money

3. **Batch Processing**
   - Process receipts in batches to minimize overhead

4. **Disable for Testing**
   ```bash
   receipt-ocr --no-llm  # Use regex-based parsing only
   ```

### Estimated Costs

Based on typical receipt processing:

| Provider | Model | Cost per Receipt | 100 Receipts |
|----------|-------|------------------|--------------|
| OpenAI | gpt-4o-mini | $0.0001 | $0.01 |
| OpenAI | gpt-4o | $0.002 | $0.20 |
| Anthropic | Haiku | $0.001 | $0.10 |
| Anthropic | Sonnet | $0.003 | $0.30 |

*Costs are approximate and vary based on receipt complexity*

## Troubleshooting

### "Provider not supported" error

Make sure you've installed the required dependencies:
```bash
pip install -e ".[anthropic]"  # For Anthropic Claude
```

### Authentication errors

Check that the appropriate environment variable is set:
```bash
# OpenAI (default)
echo $OPENAI_API_KEY

# Anthropic
echo $ANTHROPIC_API_KEY

# Azure OpenAI
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT
```

### Low extraction quality

Try upgrading to a more powerful model:
```bash
# OpenAI: mini → full
receipt-ocr --llm-provider openai --llm-model gpt-4o

# Anthropic: Haiku → Sonnet
receipt-ocr --llm-provider anthropic --llm-model claude-3-5-sonnet-20241022
```

### Slow processing

Use faster models or disable LLM:
```bash
# Fast models (default)
receipt-ocr --llm-provider openai --llm-model gpt-4o-mini
receipt-ocr --llm-provider anthropic --llm-model claude-3-5-haiku-20241022

# Disable LLM entirely
receipt-ocr --no-llm
```

## Model Selection Guide

### When to Use Each Provider

**OpenAI GPT:**
- Very cost-effective (gpt-4o-mini) - 10x cheaper than Anthropic
- Fast response times
- Good for high-volume processing
- JSON mode ensures structured output
- Default recommendation

**Anthropic Claude:**
- Best overall quality for receipt parsing
- Good at understanding complex layouts
- Handles OCR errors exceptionally well
- Worth the extra cost for difficult receipts

**Azure OpenAI:**
- Enterprise compliance needs
- Data residency requirements
- Already using Azure infrastructure
- Consistent with other Azure services

### Model Comparison

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| gpt-4o-mini | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | Default choice (best value) |
| gpt-4o | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Maximum quality |
| claude-3-5-haiku | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Complex OCR errors |
| claude-3-5-sonnet | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Difficult receipts |

## Future Providers

Support for additional providers is planned:

- **Google Gemini** - Coming soon
- **Ollama (local)** - For offline processing
- **Custom API endpoints** - Bring your own model

To request a new provider, please [open an issue](https://github.com/yourusername/receipt-ocr-pipeline/issues).
