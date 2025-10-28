"""
Categorization logic for receipts based on rules.
"""

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_rules(path: Path) -> Dict:
    """Load categorization rules from JSON file."""
    if not path.exists():
        return {"categories": [], "matchers": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def categorize(vendor: str, text: str, rules: Dict) -> Tuple[str, Optional[str]]:
    """
    Categorize a receipt based on vendor and text content.

    Args:
        vendor: Vendor name
        text: Full receipt text
        rules: Rules dictionary with format:
            {
              "categories": ["Hotel","Meals","Transportation","Misc"],
              "matchers": [
                {"name":"Hotels","any":[{"vendor_re":"HILTON|MARRIOTT|HYATT"}], "category":"Hotel"},
                {"name":"Uber/Lyft","any":[{"text_re":"UBER|LYFT"}], "category":"Transportation"}
              ]
            }

    Returns:
        Tuple of (category, matcher_name)
    """
    v = vendor or ""
    t = text or ""

    for m in rules.get("matchers", []):
        any_rules = m.get("any", [])
        all_rules = m.get("all", [])
        matched_any = False

        if any_rules:
            for rule in any_rules:
                vre = rule.get("vendor_re")
                tre = rule.get("text_re")
                ok = False
                if vre and re.search(vre, v, flags=re.IGNORECASE):
                    ok = True
                if tre and re.search(tre, t, flags=re.IGNORECASE):
                    ok = True
                if ok:
                    matched_any = True
                    break
        else:
            matched_any = True  # no 'any' means don't gate on it

        matched_all = True
        for rule in all_rules:
            vre = rule.get("vendor_re")
            tre = rule.get("text_re")
            ok = True
            if vre and not re.search(vre, v, flags=re.IGNORECASE):
                ok = False
            if tre and not re.search(tre, t, flags=re.IGNORECASE):
                ok = False
            if not ok:
                matched_all = False
                break

        if matched_any and matched_all:
            return (m.get("category") or "Uncategorized", m.get("name"))

    # fallback
    return ("Uncategorized", None)
