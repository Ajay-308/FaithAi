"""
Scripture Retrieval Layer
=========================
Retrieves relevant Bible verses from local dataset before any LLM call.
This is the grounding layer that prevents hallucination.

Architecture:
  User Query
      ↓
  retrieve_scripture_context(query)
      ↓
  Relevant verses (from local data, not from model memory)
      ↓
  Passed to LLM as grounded context

Two retrieval strategies:
  1. Direct reference lookup  — "John 3:16"  → exact verse
  2. Keyword/topic search     — "anxiety"    → relevant verses
"""

from __future__ import annotations
import json
import os
import re
from typing import Optional

# Load dataset once at import time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "bible.json")

with open(_DATA_PATH, "r", encoding="utf-8") as _f:
    _BIBLE_DATA: dict = json.load(_f)

VERSES: dict[str, dict]       = _BIBLE_DATA["verses"]
TOPIC_INDEX: dict[str, list]  = _BIBLE_DATA["topic_index"]
OLD_TESTAMENT: list[str]      = _BIBLE_DATA["books"]["old_testament"]
NEW_TESTAMENT: list[str]      = _BIBLE_DATA["books"]["new_testament"]
DEUTEROCANONICAL: list[str]   = _BIBLE_DATA["books"]["deuterocanonical"]
ALL_BOOKS: set[str]           = set(OLD_TESTAMENT + NEW_TESTAMENT + DEUTEROCANONICAL)


# Public API───────────────────────

def retrieve_scripture_context(query: str, top_k: int = 4) -> list[dict]:
    """
    Main retrieval function. Given a user query, return a list of relevant
    verse dicts:  [{ "reference": str, "text": str, "topics": list }, ...]

    Strategy:
    1. Check if query contains a direct verse reference (e.g. "John 3:16")
    2. Otherwise, score all verses by keyword overlap with query
    3. Also check topic_index for known topic words in query
    4. Return top_k unique results
    """
    results: list[tuple[float, str]] = []   # (score, reference)
    seen: set[str] = set()

    query_lower = query.lower()

    # Strategy 1: Direct reference lookup ──────────────────────────────────
    direct_ref = _extract_direct_reference(query)
    if direct_ref:
        verse = VERSES.get(direct_ref)
        if verse:
            return [{"reference": direct_ref, **verse}]
        else:
            # Reference format looks valid but not in our dataset
            return []

    # Strategy 2: Topic index lookup ────────────────────────────────────────
    for topic, refs in TOPIC_INDEX.items():
        if topic.lower() in query_lower:
            for ref in refs:
                if ref not in seen:
                    results.append((2.0, ref))   # topic hits score higher
                    seen.add(ref)

    # Strategy 3: Keyword scoring over all verses ───────────────────────────
    query_words = set(_tokenize(query_lower))
    stop_words  = {"what", "does", "the", "say", "about", "is", "are", "a", "an",
                   "in", "of", "for", "to", "and", "or", "how", "why", "me", "tell",
                   "explain", "meaning", "means", "verse", "bible", "scripture"}
    query_words -= stop_words

    for ref, data in VERSES.items():
        if ref in seen:
            continue
        score = 0.0
        verse_text_words = set(_tokenize(data["text"].lower()))
        topic_words      = set(t.lower() for t in data["topics"])

        # Word overlap with verse text
        score += len(query_words & verse_text_words) * 0.5
        # Word overlap with verse topics
        score += len(query_words & topic_words) * 1.5

        if score > 0:
            results.append((score, ref))

    # Sort by score descending, take top_k
    results.sort(key=lambda x: x[0], reverse=True)
    top_refs = [ref for _, ref in results[:top_k]]

    return [{"reference": ref, **VERSES[ref]} for ref in top_refs if ref in VERSES]


def get_verse(reference: str) -> Optional[dict]:
    """
    Direct verse lookup by exact reference string.
    Returns None if verse is not in our dataset.
    """
    return VERSES.get(reference)


def validate_book_name(book: str) -> tuple[bool, str]:
    """
    Check whether a book name exists in the Bible.
    Returns (is_valid, message).
    """
    # Exact match
    if book in ALL_BOOKS:
        testament = "Old Testament" if book in OLD_TESTAMENT else (
            "New Testament" if book in NEW_TESTAMENT else "Deuterocanonical"
        )
        return True, f"{book} is a valid Bible book ({testament})"

    # Case-insensitive match
    book_lower = book.lower()
    for b in ALL_BOOKS:
        if b.lower() == book_lower:
            return True, f"{b} is a valid Bible book"

    # Partial / fuzzy match suggestions
    suggestions = [b for b in ALL_BOOKS if book_lower in b.lower() or b.lower().startswith(book_lower[:3])]
    if suggestions:
        return False, f"'{book}' is not a Bible book. Did you mean: {', '.join(suggestions[:3])}?"

    return False, f"'{book}' does not exist in the Bible. Please check the book name."


def validate_verse_ref(book: str, chapter: int, verse: int) -> tuple[bool, str]:
    """
    Structural validation: checks whether the book exists.
    (Full chapter/verse range validation would require a complete dataset.)
    """
    valid, msg = validate_book_name(book)
    if not valid:
        return False, msg
    return True, book


def format_context_block(verses: list[dict]) -> str:
    """
    Format retrieved verses into a clean context block for the LLM prompt.
    """
    if not verses:
        return ""

    lines = ["SCRIPTURE CONTEXT (retrieved from local Bible dataset):"]
    lines.append("=" * 55)
    for v in verses:
        lines.append(f"\n{v['reference']}:")
        lines.append(f'"{v["text"]}"')
    lines.append("=" * 55)
    return "\n".join(lines)


def get_all_topics() -> list[str]:
    """Return all indexed topics."""
    return sorted(TOPIC_INDEX.keys())


# Internal helpers─────────────────

def _extract_direct_reference(query: str) -> Optional[str]:
    """
    Try to extract a verse reference like 'John 3:16' or 'Romans 8:28'
    from the query string. Returns normalized reference or None.
    """
    pattern = r'\b([1-3]?\s?[A-Za-z]+)\s+(\d+):(\d+)\b'
    match   = re.search(pattern, query)
    if not match:
        return None

    raw_book = match.group(1).strip().title()
    chapter  = match.group(2)
    verse    = match.group(3)

    # Normalize "1 John" etc.
    ref = f"{raw_book} {chapter}:{verse}"

    # Check exact
    if ref in VERSES:
        return ref

    # Try case-insensitive
    for v_ref in VERSES:
        if v_ref.lower() == ref.lower():
            return v_ref

    # Reference format is valid but may not be in our dataset — return as-is
    # Caller can decide what to do
    return ref


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z']+", text.lower())