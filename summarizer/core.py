"""Core extractive NLP logic for AI Meeting Summarizer.

Everything here is pure-Python, rule-based / statistical (regex pattern
matching + a from-scratch TextRank-style graph centrality scorer). No
external NLP libraries and no LLM API calls are used anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Sentence:
    """A single sentence extracted from the transcript."""

    text: str
    speaker: Optional[str]
    index: int  # position in the overall transcript (0-based, in order)


# ---------------------------------------------------------------------------
# Stopwords (small built-in list; no external corpus/download needed)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "about", "as", "by", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "we", "i", "you", "he", "she", "they", "them", "us",
    "our", "your", "their", "his", "her", "will", "would", "can", "could",
    "should", "shall", "may", "might", "do", "does", "did", "have", "has",
    "had", "not", "no", "yes", "up", "out", "just", "also", "there",
    "here", "who", "what", "when", "where", "why", "how", "all", "some",
    "into", "over", "than", "too", "very", "s", "t", "get", "got", "im",
}

SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .'\-]{0,40}?):\s*(.*)$")

# Split on sentence-ending punctuation followed by whitespace or end of
# string, while keeping the punctuation attached to the sentence.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=[.!?])$")

WORD_RE = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(text: str) -> List[Sentence]:
    """Parse a speaker-labeled transcript into a flat, ordered sentence list.

    Lines look like ``"Alice: I think we should ship Friday."``. Lines
    without a recognizable ``Speaker:`` prefix are still included, with
    ``speaker=None``, so the tool degrades gracefully on unlabeled text.
    """
    if not text or not text.strip():
        return []

    sentences: List[Sentence] = []
    idx = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_LINE_RE.match(line)
        if match:
            speaker = match.group(1).strip()
            body = match.group(2).strip()
        else:
            speaker = None
            body = line

        if not body:
            continue

        for piece in _split_sentences(body):
            piece = piece.strip()
            if piece:
                sentences.append(Sentence(text=piece, speaker=speaker, index=idx))
                idx += 1

    return sentences


def _split_sentences(body: str) -> List[str]:
    parts = [p for p in SENTENCE_SPLIT_RE.split(body) if p]
    return parts if parts else [body]


# ---------------------------------------------------------------------------
# Decision extraction
# ---------------------------------------------------------------------------

DECISION_PATTERNS = [
    re.compile(r"\bwe(?:'ve| have)? decided\b", re.IGNORECASE),
    re.compile(r"\bdecided to\b", re.IGNORECASE),
    re.compile(r"\bagreed to\b", re.IGNORECASE),
    re.compile(r"\bwe agree(?:d)?\b", re.IGNORECASE),
    re.compile(r"\blet'?s go with\b", re.IGNORECASE),
    re.compile(r"\bwe'?re going with\b", re.IGNORECASE),
    re.compile(r"\bwe will go with\b", re.IGNORECASE),
    re.compile(r"\bthe decision is\b", re.IGNORECASE),
    re.compile(r"\bfinal decision\b", re.IGNORECASE),
    re.compile(r"\bconsensus is\b", re.IGNORECASE),
    re.compile(r"\bit'?s settled\b", re.IGNORECASE),
    re.compile(r"\bwe'?ll (?:proceed|move forward) with\b", re.IGNORECASE),
]


def extract_decisions(sentences: List[Sentence]) -> List[Dict[str, Any]]:
    """Return sentences that match decision-indicating language patterns."""
    decisions = []
    for sent in sentences:
        for pattern in DECISION_PATTERNS:
            if pattern.search(sent.text):
                decisions.append({
                    "text": sent.text,
                    "speaker": sent.speaker,
                })
                break
    return decisions


# ---------------------------------------------------------------------------
# Due-date extraction
# ---------------------------------------------------------------------------

_WEEKDAYS = r"(?:Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:rs(?:day)?)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

DUE_DATE_PATTERNS = [
    re.compile(
        r"\bby\s+(next\s+" + _WEEKDAYS + r"|this\s+" + _WEEKDAYS + r"|"
        + _WEEKDAYS + r"|next\s+week|end\s+of\s+(?:day|week|month)|"
        r"tomorrow|today|\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
        + _MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(next\s+week|next\s+month|end\s+of\s+(?:day|week|month))\b", re.IGNORECASE),
    re.compile(r"\bdue\s+(next\s+" + _WEEKDAYS + r"|" + _WEEKDAYS + r"|tomorrow|today|next\s+week)", re.IGNORECASE),
    re.compile(r"\bbefore\s+(next\s+" + _WEEKDAYS + r"|" + _WEEKDAYS + r"|the\s+meeting|end\s+of\s+(?:day|week|month))", re.IGNORECASE),
]


def extract_due_date(text: str) -> Optional[str]:
    """Return the first due-date phrase found in ``text``, or ``None``."""
    for pattern in DUE_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            if match.groups():
                return match.group(1).strip()
            return match.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# Action item extraction
# ---------------------------------------------------------------------------

# "Bob will follow up with legal" / "Bob will send the deck by Friday"
_ASSIGNEE_WILL_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\s+will\s+(.+)")

# "Alice to send the deck by Monday" (explicit assignment, not "going to")
_ASSIGNEE_TO_RE = re.compile(r"^([A-Z][a-zA-Z]+)\s+to\s+(?!be\b)([a-z].+)")

# "Bob, can you send..." / "Bob, please review..."
_ADDRESS_RE = re.compile(r"\b([A-Z][a-zA-Z]+),?\s+(?:can you|could you|please)\s+(.+)", re.IGNORECASE)

# First-person self-commitment: "I'll send the report by Friday."
_SELF_COMMIT_RE = re.compile(r"\bI(?:'ll| will)\s+(.+)")

# Explicit "Action item:" / "Action item for X:" markers
_ACTION_MARKER_RE = re.compile(r"\baction item(?:s)?\s*(?:for\s+([A-Z][a-zA-Z]+))?\s*[:\-]\s*(.+)", re.IGNORECASE)

# Common sentence-initial words that are capitalized but are NOT names, so
# they must never be accepted as an assignee by the "<Word> will/to ..."
# patterns (e.g. "Glad to hear it." should not extract "Glad" as a person).
NON_NAME_WORDS = {
    "i", "we", "they", "it", "this", "that", "there", "here",
    "glad", "sure", "yes", "no", "thanks", "thank", "okay", "ok",
    "great", "nice", "sounds", "sorry", "please", "nothing",
    "something", "everything", "anything", "let", "lets", "alright",
    "well", "so", "now", "also", "still", "just", "actually",
    "honestly", "basically", "totally", "definitely", "absolutely",
    "before", "after", "since", "until", "once",
}


def _is_plausible_name(word: str) -> bool:
    return bool(word) and word.lower() not in NON_NAME_WORDS


# Bare imperative verbs that indicate a task even without an explicit "will"
IMPERATIVE_VERBS = (
    "send", "follow up", "prepare", "reach out", "schedule", "update",
    "review", "finalize", "create", "draft", "share", "set up", "email",
    "call", "book", "coordinate", "confirm", "circulate", "put together",
    "look into", "check", "file", "submit",
)
_IMPERATIVE_RE = re.compile(
    r"^(?:" + "|".join(re.escape(v) for v in IMPERATIVE_VERBS) + r")\b",
    re.IGNORECASE,
)


def extract_action_items(sentences: List[Sentence]) -> List[Dict[str, Any]]:
    """Return action items as dicts with ``text``, ``assignee``, ``due_date``."""
    items: List[Dict[str, Any]] = []

    for sent in sentences:
        text = sent.text
        assignee = None
        matched = False

        marker_match = _ACTION_MARKER_RE.search(text)
        will_match = _ASSIGNEE_WILL_RE.search(text)
        to_match = _ASSIGNEE_TO_RE.search(text)
        address_match = _ADDRESS_RE.search(text)
        self_match = _SELF_COMMIT_RE.search(text)

        if marker_match:
            assignee = marker_match.group(1) or sent.speaker
            matched = True
        elif will_match and _is_plausible_name(will_match.group(1)):
            assignee = will_match.group(1)
            matched = True
        elif to_match and _is_plausible_name(to_match.group(1)):
            assignee = to_match.group(1)
            matched = True
        elif address_match and _is_plausible_name(address_match.group(1)):
            assignee = address_match.group(1)
            matched = True
        elif self_match:
            assignee = sent.speaker
            matched = True
        elif _IMPERATIVE_RE.search(text):
            # Bare imperative ("Send the notes to the team by Friday.")
            # Assign to a mentioned name if present, else the speaker.
            name_in_text = re.search(r"\b([A-Z][a-zA-Z]+)\b", text)
            if name_in_text and _is_plausible_name(name_in_text.group(1)):
                assignee = name_in_text.group(1)
            else:
                assignee = sent.speaker
            matched = True

        if not matched:
            continue

        due_date = extract_due_date(text)
        items.append({
            "text": text,
            "assignee": assignee,
            "due_date": due_date,
            "speaker": sent.speaker,
        })

    return items


# ---------------------------------------------------------------------------
# Extractive summary: TextRank-style graph centrality, implemented from
# scratch (no networkx / sklearn / nltk).
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS and len(w) > 1]


def _sentence_similarity(a_words: List[str], b_words: List[str]) -> float:
    """Classic TextRank sentence similarity: normalized word overlap."""
    if not a_words or not b_words:
        return 0.0
    set_a, set_b = set(a_words), set(b_words)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0
    import math
    denom = math.log(len(set_a) + 1) + math.log(len(set_b) + 1)
    if denom == 0:
        return 0.0
    return overlap / denom


def _pagerank(matrix: List[List[float]], damping: float = 0.85,
              max_iter: int = 100, tol: float = 1e-6) -> List[float]:
    n = len(matrix)
    if n == 0:
        return []
    scores = [1.0 / n] * n

    # Precompute row sums (out-degree weight) for normalization.
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]

    for _ in range(max_iter):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if matrix[j][i] > 0 and col_sums[j] > 0:
                    rank_sum += matrix[j][i] / col_sums[j] * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum

        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < tol:
            break

    return scores


def score_sentences(sentences: List[Sentence]) -> List[float]:
    """Score each sentence by TextRank-style graph centrality."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    tokenized = [_tokenize(s.text) for s in sentences]
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i][j] = _sentence_similarity(tokenized[i], tokenized[j])

    return _pagerank(matrix)


def summarize(sentences: List[Sentence], max_sentences: int = 4) -> str:
    """Produce a short extractive summary via TextRank-style centrality.

    Picks the top-scoring sentences (by graph centrality) and returns them
    joined in their original transcript order, so the summary reads
    coherently rather than as a jumbled ranking.
    """
    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        chosen = sorted(sentences, key=lambda s: s.index)
        return " ".join(s.text for s in chosen)

    scores = score_sentences(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:max_sentences])
    chosen = [sentences[i] for i in top_indices]
    return " ".join(s.text for s in chosen)


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def analyze_transcript(text: str, summary_sentences: int = 4) -> Dict[str, Any]:
    """Run the full pipeline over a raw transcript string."""
    sentences = parse_transcript(text)
    return {
        "decisions": extract_decisions(sentences),
        "action_items": extract_action_items(sentences),
        "summary": summarize(sentences, max_sentences=summary_sentences),
        "sentence_count": len(sentences),
    }
