"""Unit tests for the AI Meeting Summarizer's extractive NLP pipeline."""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from summarizer.core import (
    Sentence,
    parse_transcript,
    extract_decisions,
    extract_action_items,
    extract_due_date,
    summarize,
    analyze_transcript,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestDecisionExtraction(unittest.TestCase):
    def test_decision_extraction_known_phrasing(self):
        transcript = (
            "Alice: We decided to ship the feature on Friday.\n"
            "Bob: Sounds good, I agree.\n"
            "Carol: Let's go with the blue design for the homepage.\n"
            "Alice: The weather is nice today.\n"
        )
        sentences = parse_transcript(transcript)
        decisions = extract_decisions(sentences)
        texts = [d["text"] for d in decisions]

        self.assertEqual(len(decisions), 2)
        self.assertTrue(any("decided to ship" in t for t in texts))
        self.assertTrue(any("go with the blue design" in t for t in texts))
        # Non-decision sentence must not be flagged.
        self.assertFalse(any("weather is nice" in t for t in texts))
        # Speaker attribution should be captured.
        self.assertEqual(decisions[0]["speaker"], "Alice")


class TestActionItemExtraction(unittest.TestCase):
    def test_action_item_extraction_with_correct_assignee(self):
        transcript = (
            "Alice: Let's review the plan.\n"
            "Bob: Bob will follow up with legal about the contract.\n"
        )
        sentences = parse_transcript(transcript)
        items = extract_action_items(sentences)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["assignee"], "Bob")
        self.assertIn("follow up with legal", items[0]["text"])

    def test_self_commitment_action_item_uses_speaker_as_assignee(self):
        transcript = "Carol: I'll send the deck to the team by Friday.\n"
        sentences = parse_transcript(transcript)
        items = extract_action_items(sentences)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["assignee"], "Carol")
        self.assertEqual(items[0]["due_date"], "Friday")

    def test_addressed_action_item_assignee(self):
        transcript = "Alice: Dave, can you finalize the mockups by Monday?\n"
        sentences = parse_transcript(transcript)
        items = extract_action_items(sentences)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["assignee"], "Dave")


class TestDueDateExtraction(unittest.TestCase):
    def test_due_date_phrase_extraction(self):
        cases = {
            "Bob will send the report by Friday.": "Friday",
            "We need this done by next Monday.": "next Monday",
            "Let's finish this next week.": "next week",
            "The task is due tomorrow.": "tomorrow",
            "No deadline mentioned here at all.": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_due_date(text), expected)


class TestExtractiveSummary(unittest.TestCase):
    def test_summary_picks_genuinely_central_sentence(self):
        # S1 and S2 share heavy vocabulary overlap ("authentication",
        # "system", "single", "sign") with each other and with S0, while
        # S3/S4 are unrelated filler. The centrality algorithm should
        # surface one of the two hub sentences, never the filler.
        sentences = [
            Sentence("The team reviewed the new authentication system today.", "Alice", 0),
            Sentence("The authentication system now supports single sign on and two factor login.", "Bob", 1),
            Sentence("Single sign on adoption has increased significantly since the authentication system update.", "Carol", 2),
            Sentence("Lunch was catered from the new restaurant downtown.", "Bob", 3),
            Sentence("Parking was difficult to find this morning.", "Carol", 4),
        ]
        top_summary = summarize(sentences, max_sentences=1)
        self.assertIn(
            top_summary,
            [sentences[1].text, sentences[2].text],
            "Expected the summary to pick one of the two densely-connected "
            "'hub' sentences, not an unrelated filler sentence.",
        )
        self.assertNotEqual(top_summary, sentences[3].text)
        self.assertNotEqual(top_summary, sentences[4].text)


class TestEdgeCases(unittest.TestCase):
    def test_empty_transcript_handled_gracefully(self):
        result = analyze_transcript("")
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["action_items"], [])
        self.assertEqual(result["summary"], "")

    def test_very_short_transcript_handled_gracefully(self):
        result = analyze_transcript("Alice: Hi everyone, thanks for joining.")
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["action_items"], [])
        self.assertIn("Hi everyone", result["summary"])

    def test_no_decisions_or_action_items_returns_empty_lists(self):
        transcript = (
            "Alice: How was your weekend?\n"
            "Bob: Pretty relaxing, thanks for asking.\n"
            "Alice: Glad to hear it.\n"
        )
        sentences = parse_transcript(transcript)
        self.assertEqual(extract_decisions(sentences), [])
        self.assertEqual(extract_action_items(sentences), [])


class TestParsing(unittest.TestCase):
    def test_speaker_attribution_parsed_correctly(self):
        transcript = "Alice: I think we should ship Friday. It's an easy win."
        sentences = parse_transcript(transcript)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(all(s.speaker == "Alice" for s in sentences))


class TestCLI(unittest.TestCase):
    def test_cli_runs_on_sample_transcript(self):
        sample = os.path.join(REPO_ROOT, "samples", "product_launch.txt")
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "summarize.py"), sample],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Key Decisions", result.stdout)
        self.assertIn("Action Items", result.stdout)
        self.assertIn("Summary", result.stdout)


if __name__ == "__main__":
    unittest.main()
