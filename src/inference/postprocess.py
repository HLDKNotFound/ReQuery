from __future__ import annotations
import re
from typing import List

def clean_query_text(text: str) -> str:
    """Clean and strip special artifacts from model generation."""
    if not text:
        return ""
    # Strip ChatML tags and common prompt artifacts
    text = re.sub(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>", "", text)
    text = re.sub(r"^(Assistant|Query|Rewrite|Search Query)[:\s\n]+", "", text, flags=re.IGNORECASE)
    # Strip leading numbers or bullets like '1. ', '- '
    text = re.sub(r"^(\d+[\.\)]\s*|[\-\*]\s*)", "", text)
    # Strip surrounding quotation marks
    text = text.strip().strip('"\'')
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def deduplicate_queries(queries: List[str], preserve_order: bool = True) -> List[str]:
    """Deduplicate queries case-insensitively while preserving generation order."""
    seen = set()
    deduped = []
    for q in queries:
        cleaned = clean_query_text(q)
        if not cleaned:
            continue
        lower_q = cleaned.lower()
        if lower_q not in seen:
            seen.add(lower_q)
            deduped.append(cleaned)
    return deduped
