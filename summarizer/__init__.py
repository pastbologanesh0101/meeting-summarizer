"""AI Meeting Summarizer - extractive, rule-based meeting transcript analysis.

No LLM calls. Pure stdlib Python.
"""

from .core import (
    Sentence,
    parse_transcript,
    extract_decisions,
    extract_action_items,
    extract_due_date,
    summarize,
    analyze_transcript,
)

__all__ = [
    "Sentence",
    "parse_transcript",
    "extract_decisions",
    "extract_action_items",
    "extract_due_date",
    "summarize",
    "analyze_transcript",
]
