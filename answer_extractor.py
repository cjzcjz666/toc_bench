#!/usr/bin/env python3
"""
TOC-Bench Answer Extractor
==========================
Per-format extraction from free-form model output. Robust to:
  - leading/trailing whitespace and punctuation
  - "Answer: X" prefixes
  - "The answer is X" verbose responses
  - markdown formatting (**X**, "X", etc.)

Returns a normalized answer string, or None if extraction failed.
None is treated as INCORRECT in scoring (not abstain).

Per format:
  mcq_4         → 'A' / 'B' / 'C' / 'D' / None
  sp            → 'A' / 'B' / None
  numerical     → '2' / '3' / '4' / '5 or more' / None
  ordering_3    → list of 3 unique letters, e.g. ['B','A','C'] / None
  ordering_4    → list of 4 unique letters / None

Usage:
    from answer_extractor import extract
    pred = extract(raw_text, format='mcq_4')
"""

import re
from typing import Optional, Union, List


# ============================================================================
# MCQ_4 / SP — single letter
# ============================================================================

_LETTER_PREFIX = re.compile(
    r"^[\s\*\"'`(\[]*(?:answer\s*[:\-]\s*|the\s+answer\s+is\s+|option\s+)?[\s\*\"'`(\[]*"
    r"([A-D])\b",
    re.IGNORECASE,
)
_FIRST_LETTER = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_LAST_STANDALONE_LETTER = re.compile(r"\b([A-D])\b[\s\.\)\]\"'`]*$", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"(?is)<think>.*?</think>")


def _final_answer_region(text: str) -> str:
    """Prefer concise answer tail after </think> when present."""
    if not text:
        return ""
    text = text.strip()
    if "</think>" in text.lower():
        tail = re.split(r"(?i)</think>", text, maxsplit=1)[-1].strip()
        if tail:
            return tail
    # If there is only a think block and no closing tag, strip it anyway.
    return _THINK_BLOCK.sub(" ", text).strip()


def _extract_letter(text: str, allowed: str) -> Optional[str]:
    """Extract first letter from `allowed` (e.g. 'AB' or 'ABCD')."""
    if not text:
        return None
    text = _final_answer_region(text)

    # Try prefix-style: "Answer: X" / "The answer is X" / leading "X"
    m = _LETTER_PREFIX.match(text)
    if m:
        letter = m.group(1).upper()
        if letter in allowed:
            return letter

    # Prefer a clean final token like "... \n\nB" or "Answer: C."
    m = _LAST_STANDALONE_LETTER.search(text)
    if m:
        letter = m.group(1).upper()
        if letter in allowed:
            return letter

    # Try any standalone letter in the response
    for m in _FIRST_LETTER.finditer(text):
        letter = m.group(1).upper()
        if letter in allowed:
            return letter

    return None


def extract_mcq_4(text: str) -> Optional[str]:
    return _extract_letter(text, "ABCD")


def extract_sp(text: str) -> Optional[str]:
    return _extract_letter(text, "AB")


# ============================================================================
# Numerical — "2" / "3" / "4" / "5 or more"
# ============================================================================

_FIVE_OR_MORE = re.compile(r"\b5\s*(?:or\s*more|\+|more)\b", re.IGNORECASE)
_PLAIN_INT = re.compile(r"\b([0-9]+)\b")


def extract_numerical(text: str) -> Optional[str]:
    if not text:
        return None
    text = _final_answer_region(text)

    # First check explicit "5 or more" phrasing
    if _FIVE_OR_MORE.search(text):
        return "5 or more"

    # Find first integer
    m = _PLAIN_INT.search(text)
    if not m:
        return None
    n = int(m.group(1))
    if n <= 1:
        return None  # outside answer space
    if n in (2, 3, 4):
        return str(n)
    if n >= 5:
        return "5 or more"
    return None


# ============================================================================
# Ordering — list of unique letters
# ============================================================================

_ORDERING_LETTERS = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def extract_ordering(text: str, k: int) -> Optional[List[str]]:
    """Pull out the first k unique letters in order. Letters past k are
    ignored. Returns None if fewer than k unique letters found."""
    if not text:
        return None
    text = _final_answer_region(text)
    seen_in_order = []
    seen = set()
    for m in _ORDERING_LETTERS.finditer(text):
        letter = m.group(1).upper()
        if letter in seen:
            continue
        seen.add(letter)
        seen_in_order.append(letter)
        if len(seen_in_order) == k:
            return seen_in_order
    return None


# ============================================================================
# Unified entry point
# ============================================================================

def extract(text: str, format: str) -> Optional[Union[str, List[str]]]:
    """Dispatch by format name.

    format ∈ {'mcq_4', 'sp', 'numerical', 'ordering_3', 'ordering_4'}.
    Returns string for mcq/sp/numerical, list for ordering, None if invalid.
    """
    if format == "mcq_4":
        return extract_mcq_4(text)
    if format == "sp":
        return extract_sp(text)
    if format == "numerical":
        return extract_numerical(text)
    if format == "ordering_3":
        return extract_ordering(text, 3)
    if format == "ordering_4":
        return extract_ordering(text, 4)
    raise ValueError(f"Unknown format: {format}")


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    cases = [
        # (text, format, expected)
        ("A", "mcq_4", "A"),
        ("D.", "mcq_4", "D"),
        ("Answer: B", "mcq_4", "B"),
        ("**C**", "mcq_4", "C"),
        ("The answer is C.", "mcq_4", "C"),
        ("Based on the video, I think B is correct.", "mcq_4", "B"),
        ("E", "mcq_4", None),
        ("", "mcq_4", None),

        ("A", "sp", "A"),
        ("answer: B", "sp", "B"),
        ("D", "sp", None),  # D not in AB

        ("3", "numerical", "3"),
        ("The answer is 4.", "numerical", "4"),
        ("5 or more", "numerical", "5 or more"),
        ("5+", "numerical", "5 or more"),
        ("It happens 7 times", "numerical", "5 or more"),
        ("twice", "numerical", None),
        ("", "numerical", None),

        ("B, A, C", "ordering_3", ["B", "A", "C"]),
        ("B,A,C", "ordering_3", ["B", "A", "C"]),
        ("D, B, A, C", "ordering_4", ["D", "B", "A", "C"]),
        ("Order: A then B then C then D", "ordering_4", ["A", "B", "C", "D"]),
        ("<think>reasoning with A B C D</think>\n\nB, A, C", "ordering_3", ["B", "A", "C"]),
        ("<think>long chain mentioning D C B A</think>\n\nD, B, A, C", "ordering_4", ["D", "B", "A", "C"]),
        ("A, B", "ordering_3", None),  # only 2 letters
    ]
    fail = 0
    for text, fmt, expected in cases:
        got = extract(text, fmt)
        ok = got == expected
        mark = "✓" if ok else "✗"
        if not ok:
            fail += 1
        print(f"  {mark} extract({text!r:<40s}, {fmt}) = {got!r}  (expected {expected!r})")
    print(f"\n{fail} failures of {len(cases)}")