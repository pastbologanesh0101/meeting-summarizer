#!/usr/bin/env python3
"""CLI entry point for AI Meeting Summarizer.

Usage:
    python summarize.py transcript.txt
"""

import argparse
import sys

from summarizer.core import analyze_transcript


def format_report(result: dict, transcript_name: str) -> str:
    lines = []
    lines.append(f"=== Meeting Summary: {transcript_name} ===\n")

    lines.append("-- Key Decisions --")
    if result["decisions"]:
        for d in result["decisions"]:
            speaker = f"[{d['speaker']}] " if d["speaker"] else ""
            lines.append(f"  - {speaker}{d['text']}")
    else:
        lines.append("  (none found)")
    lines.append("")

    lines.append("-- Action Items --")
    if result["action_items"]:
        for item in result["action_items"]:
            assignee = item["assignee"] or "Unassigned"
            due = f" (due: {item['due_date']})" if item["due_date"] else ""
            lines.append(f"  - [{assignee}]{due} {item['text']}")
    else:
        lines.append("  (none found)")
    lines.append("")

    lines.append("-- Summary --")
    lines.append(result["summary"] if result["summary"] else "(transcript too short to summarize)")

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extractive, rule-based meeting transcript summarizer (no LLM calls).",
    )
    parser.add_argument("transcript", help="Path to a plain-text meeting transcript.")
    parser.add_argument(
        "--sentences", type=int, default=4,
        help="Target number of sentences in the summary (default: 4).",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.transcript, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"Error reading {args.transcript}: {exc}", file=sys.stderr)
        return 1

    result = analyze_transcript(text, summary_sentences=args.sentences)
    print(format_report(result, args.transcript))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
