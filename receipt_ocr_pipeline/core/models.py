"""
Data models for receipt processing.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ReceiptEntry:
    """Represents a parsed receipt entry."""
    date: Optional[str]
    vendor: Optional[str]
    amount: Optional[float]
    category: str
    matcher: Optional[str]
    notes: Optional[str]
    source_file: str
    sha1: str

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
