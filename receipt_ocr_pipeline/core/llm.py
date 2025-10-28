"""
LLM-based extraction supporting multiple providers (Anthropic, OpenAI, etc.).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure-openai"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.AZURE_OPENAI: "gpt-4o-mini",
}

# Lazy import clients
_clients = {}


def _get_anthropic_client():
    """Get or create Anthropic client (lazy initialization)."""
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    return _clients["anthropic"]


def _get_openai_client():
    """Get or create OpenAI client (lazy initialization)."""
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    return _clients["openai"]


def _get_azure_openai_client():
    """Get or create Azure OpenAI client (lazy initialization)."""
    if "azure" not in _clients:
        import openai
        _clients["azure"] = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    return _clients["azure"]


def _call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic API."""
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def _call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API."""
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()


def _call_azure_openai(prompt: str, model: str) -> str:
    """Call Azure OpenAI API."""
    client = _get_azure_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip()


def extract_with_llm(ocr_text: str, categories: List[str],
                     use_llm: bool = True,
                     file_hash: Optional[str] = None,
                     cache_path: Optional[Path] = None,
                     provider: str = "openai",
                     model: Optional[str] = None) -> Dict:
    """
    Extract vendor name and category from receipt text using LLM.

    Args:
        ocr_text: Full OCR text from receipt
        categories: List of available categories
        use_llm: Whether to use LLM (can be disabled for testing/fallback)
        file_hash: SHA1 hash of file for caching (optional)
        cache_path: Path to cache database (optional)
        provider: LLM provider to use ("openai", "anthropic", "azure-openai") - default: openai
        model: Model name (uses default for provider if not specified)

    Returns:
        Dict with keys: vendor, date, category, confidence (0.0-1.0), reasoning, cached (bool)
    """
    # Import database functions for caching
    from .database import get_llm_cache, save_llm_cache

    if not use_llm:
        return {
            "vendor": None,
            "date": None,
            "category": "Uncategorized",
            "confidence": 0.0,
            "reasoning": "LLM disabled",
            "cached": False
        }

    # Check cache first
    if file_hash and cache_path:
        cached_result = get_llm_cache(cache_path, file_hash)
        if cached_result:
            return cached_result

    try:
        # Determine model to use
        if model is None:
            model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS[LLMProvider.ANTHROPIC])

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

        # Call appropriate provider
        if provider == LLMProvider.ANTHROPIC:
            response_text = _call_anthropic(prompt, model)
        elif provider == LLMProvider.OPENAI:
            response_text = _call_openai(prompt, model)
        elif provider == LLMProvider.AZURE_OPENAI:
            response_text = _call_azure_openai(prompt, model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

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
