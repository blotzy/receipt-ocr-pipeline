"""
Parsers for extracting information from receipt text.
"""

import re
import datetime as dt
from typing import Optional, List, Dict

from .utils import DATE_PATTERNS, AMOUNT_PATTERNS, normalize_amount


def parse_date(text: str) -> Optional[str]:
    """Extract date from receipt text."""
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            g = m.groups()
            # Try to interpret
            try:
                if len(g) == 3:
                    # pattern-based normalization
                    if pat.startswith(r"\b(\d{4})"):
                        y, mo, d = int(g[0]), int(g[1]), int(g[2])
                    elif pat.startswith(r"\b(\d{1,2})"):
                        # Heuristic: if last group is 4 digits -> year
                        mo, d, y = int(g[0]), int(g[1]), int(g[2])
                        if y < 100:  # YY -> 20YY
                            y += 2000
                        # If looks like DD/MM, swap if mo > 12
                        if mo > 12 and d <= 12:
                            mo, d = d, mo
                    else:
                        # Month name
                        mn, d, y = g[0], int(g[1]), int(g[2])
                        mo = dt.datetime.strptime(mn[:3], "%b").month
                    dtobj = dt.date(y, mo, d)
                    return dtobj.isoformat()
            except Exception:
                continue
    return None


def parse_amount(text: str) -> Optional[float]:
    """
    Extract the total amount from receipt text.
    Tries multiple patterns, prioritizing labeled amounts.
    """
    candidates = []

    for pat in AMOUNT_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            val = normalize_amount(m.group(1))
            if val and val > 0:
                # Track position to prioritize amounts near end of receipt
                candidates.append((val, m.start()))

    if not candidates:
        return None

    # If we have labeled amounts (from first 2 patterns), prefer those
    labeled_amounts = [c for c in candidates[:5] if c[0]]  # First few are from labeled patterns
    if labeled_amounts:
        # Return the largest labeled amount (usually the total)
        return max(labeled_amounts, key=lambda x: x[0])[0]

    # Otherwise, prefer amounts that appear later in the text (near bottom of receipt)
    # and are larger (likely the total rather than subtotals)
    if candidates:
        # Sort by position (later is better) and value (larger is better)
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return candidates[0][0]

    return None


def parse_vendor(text: str, blacklist: Optional[List[str]] = None,
                 hints: Optional[Dict[str, str]] = None,
                 rules: Optional[Dict] = None) -> Optional[str]:
    """
    Extract vendor name from receipt text with improved heuristics.

    Args:
        text: OCR'd receipt text
        blacklist: List of strings to exclude (e.g., customer names)
        hints: Dict mapping text patterns to vendor names for manual override
        rules: Full rules dict with matchers that may contain vendor_extract patterns

    Returns:
        Vendor name or None
    """
    blacklist = blacklist or []
    blacklist_lower = [b.lower() for b in blacklist]
    hints = hints or {}
    rules = rules or {}

    # First, check vendor hints for manual overrides
    text_lower = text.lower()
    for pattern, vendor_name in hints.items():
        if pattern.lower() in text_lower:
            return vendor_name

    # Try to extract using patterns from matchers
    for matcher in rules.get("matchers", []):
        vendor_extract_patterns = matcher.get("vendor_extract", [])
        for pattern_obj in vendor_extract_patterns:
            pattern = pattern_obj.get("pattern")
            if pattern:
                match = re.search(pattern, text)
                if match:
                    extracted = match.group(0)
                    # Check if it's not in blacklist
                    if not any(bl in extracted.lower() for bl in blacklist_lower):
                        return extracted

    # Common patterns to skip (customer-related text, receipt metadata)
    skip_patterns = [
        r'^(customer|cardholder|card\s*holder|member|account|name)[\s:]*',
        r'^(thank\s*you|thanks|receipt|invoice|bill|order|ticket)',
        r'^(date|time|server|cashier|clerk|sold\s*to|ship\s*to)',
        r'^\*+\s*$',  # Lines that are just asterisks
        r'^-+\s*$',   # Lines that are just dashes
        r'^=+\s*$',   # Lines that are just equals
        r'^[0-9\s\-\./]+$',  # Lines with only numbers and punctuation
        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates at start of line
        r'^(visa|mastercard|amex|discover|debit|credit)',  # Card types
    ]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = []
    for idx, ln in enumerate(lines[:20]):  # Check first 20 lines
        if len(ln) < 3 or len(ln) > 80:
            continue

        # Skip if matches any skip pattern
        skip = False
        for pattern in skip_patterns:
            if re.search(pattern, ln, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        # Skip if in blacklist
        if any(bl in ln.lower() for bl in blacklist_lower):
            continue

        letters = sum(c.isalpha() for c in ln)
        digits = sum(c.isdigit() for c in ln)
        spaces = ln.count(' ')

        # Must have mostly letters, not too many digits or spaces
        if letters >= 3 and digits < max(2, letters // 3) and spaces < 5:
            # Boost score for lines near the top
            score = 100 - idx * 3  # Earlier lines get higher scores

            # Boost for all-caps (common for business names)
            if ln.upper() == ln and 3 <= len(ln) <= 40:
                score += 50

            # Boost for moderate length (sweet spot for business names)
            if 5 <= len(ln) <= 30:
                score += 20

            # Penalty for very long lines
            if len(ln) > 50:
                score -= 30

            # Penalty for mixed case with lots of spaces (likely address or description)
            if spaces > 2:
                score -= 15

            candidates.append((score, ln))

    if candidates:
        # Sort by score and return the best candidate
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][1]

    # Last resort: return first non-blacklisted line
    for ln in lines[:5]:
        if len(ln) >= 3 and not any(bl in ln.lower() for bl in blacklist_lower):
            skip = False
            for pattern in skip_patterns:
                if re.search(pattern, ln, re.IGNORECASE):
                    skip = True
                    break
            if not skip:
                return ln

    return lines[0] if lines else None
