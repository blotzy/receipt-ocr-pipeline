# Environment Variable Configuration

The receipt OCR pipeline supports configuration via environment variables, making it easy to set up consistent environments without repetitive CLI arguments.

## Supported Environment Variables

### LLM Configuration

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `LLM_PROVIDER` | LLM provider to use | `openai`, `anthropic`, `azure-openai` | `openai` |
| `LLM_MODEL` | Specific model to use | `gpt-4o`, `claude-3-5-sonnet-20241022` | Provider default |

### API Keys

| Variable | Description | Required For |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI provider (default) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Anthropic provider |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Azure OpenAI provider |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL | Azure OpenAI provider |
| `AZURE_OPENAI_API_VERSION` | Azure API version | Azure OpenAI provider |

## Priority Order

Configuration is resolved in this order (highest to lowest priority):

1. **CLI Arguments** - `--llm-provider`, `--llm-model` (overrides everything)
2. **Environment Variables** - `LLM_PROVIDER`, `LLM_MODEL`
3. **Built-in Defaults** - OpenAI with gpt-4o-mini

## Usage Examples

### Basic Setup

```bash
# Set up for OpenAI (default provider)
export OPENAI_API_KEY=sk-your-key-here
receipt-ocr  # Uses OpenAI with gpt-4o-mini
```

### Custom Provider

```bash
# Use Anthropic instead
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_API_KEY=sk-ant-your-key-here
receipt-ocr  # Uses Anthropic Sonnet
```

### Override with CLI

```bash
# Environment says Anthropic, but CLI overrides to OpenAI
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-key
receipt-ocr --llm-provider openai
# Result: Uses OpenAI (CLI wins)
```

### Azure OpenAI Setup

```bash
export LLM_PROVIDER=azure-openai
export LLM_MODEL=your-deployment-name
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
receipt-ocr
```

## Configuration Files

### Using .envrc (direnv)

Create a `.envrc` file in your project directory:

```bash
#!/bin/bash

# API Keys
export OPENAI_API_KEY=sk-your-key-here
# export ANTHROPIC_API_KEY=sk-ant-your-key-here

# LLM Configuration
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini

# Auto-loaded by direnv when you cd into the directory
```

Then run:
```bash
direnv allow
cd .  # Reloads environment
receipt-ocr  # Automatically configured!
```

### Using .env file (python-dotenv)

If you want to use a `.env` file instead:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

Load it before running:
```bash
# Using python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"

# Or export manually
export $(cat .env | xargs)
receipt-ocr
```

### Shell Profile (~/.bashrc or ~/.zshrc)

For permanent configuration:

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY=sk-your-key-here
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

## Use Cases

### 1. Development vs Production

```bash
# Development (.envrc.dev)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini  # Cheaper for testing

# Production (.envrc.prod)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o  # Higher quality
```

### 2. Team Consistency

Share a `.envrc.template` with your team:

```bash
# .envrc.template
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-3-5-haiku-20241022
export ANTHROPIC_API_KEY=your-team-key-here
```

Everyone uses the same configuration automatically!

### 3. CI/CD Pipeline

```yaml
# GitHub Actions
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  LLM_PROVIDER: openai
  LLM_MODEL: gpt-4o-mini

steps:
  - name: Process receipts
    run: receipt-ocr --incoming ./receipts
```

### 4. Docker Container

```dockerfile
# Dockerfile
ENV LLM_PROVIDER=openai
ENV LLM_MODEL=gpt-4o-mini

# docker-compose.yml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - LLM_PROVIDER=openai
  - LLM_MODEL=gpt-4o-mini
```

## Verification

Check which configuration is active:

```bash
# The tool will print the active configuration
receipt-ocr -v

# Output will show:
# [INFO] LLM: openai (gpt-4o-mini) [from LLM_PROVIDER env]
```

## Security Best Practices

1. **Never commit API keys** to version control
   ```bash
   # Add to .gitignore
   echo ".envrc" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use .envrc.template** for sharing structure
   ```bash
   # .envrc.template (safe to commit)
   export OPENAI_API_KEY=your-key-here
   export LLM_PROVIDER=openai
   ```

3. **Use secrets management** in production
   - AWS Secrets Manager
   - HashiCorp Vault
   - GitHub Secrets
   - Azure Key Vault

4. **Rotate keys regularly** and use separate keys for dev/prod

## Troubleshooting

### "No API key found"

```bash
# Check if key is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY=sk-your-key
```

### "Invalid provider"

```bash
# Check provider value
echo $LLM_PROVIDER

# Must be one of: openai, anthropic, azure-openai
export LLM_PROVIDER=openai
```

### "Environment variable not working"

```bash
# Make sure it's exported (not just set)
export LLM_PROVIDER=openai  # ✅ Correct
LLM_PROVIDER=openai         # ❌ Won't work

# Verify it's accessible
env | grep LLM_PROVIDER
```

### "CLI argument not overriding"

CLI arguments ALWAYS override environment variables. If it's not working, check:

```bash
# This should use OpenAI regardless of environment
receipt-ocr --llm-provider openai

# Debug: run with verbose to see configuration
receipt-ocr --llm-provider openai -v
```

## Complete Example

Here's a complete working setup:

```bash
# 1. Set up API key
export OPENAI_API_KEY=sk-your-actual-key-here

# 2. Configure LLM (optional - has defaults)
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini

# 3. Run - configuration is automatic!
receipt-ocr --incoming ./receipts --output ./processed

# 4. Override for a specific run
receipt-ocr --llm-provider anthropic --incoming ./receipts

# 5. Disable LLM entirely (use regex only)
receipt-ocr --no-llm --incoming ./receipts
```

## Reference

For more information, see:
- [README.md](README.md) - General usage
- [LLM_SUPPORT.md](LLM_SUPPORT.md) - Detailed LLM provider guide
- [PACKAGE_STRUCTURE.md](PACKAGE_STRUCTURE.md) - Package structure overview
