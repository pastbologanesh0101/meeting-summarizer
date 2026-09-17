# AI Meeting Summarizer

A small, pure-Python tool that turns a speaker-labeled meeting transcript into:

- **Key decisions** — sentences matching decision-indicating language ("we decided", "agreed to", "let's go with", ...)
- **Action items** — tasks with an assignee (from explicit "X will ...", "X to ...", "Name, can you ...", or first-person "I'll ..." self-commitments) and, when present, a due-date phrase ("by Friday", "next week", "due tomorrow", ...)
- **A short summary** — 3-5 sentences picked by an extractive, graph-centrality (TextRank-style) sentence scorer

**No LLM calls, no external NLP libraries.** Everything is regex pattern matching plus a from-scratch graph-centrality algorithm (a simplified PageRank over sentence-similarity edges), implemented with the Python standard library only.

## How it works

### 1. Parsing
`summarizer/core.py:parse_transcript` splits each `Speaker: text` line into individual sentences (on `.`/`!`/`?`), tagging every sentence with its speaker and position in the transcript. Unlabeled lines are still processed, with `speaker=None`, so the tool degrades gracefully on plain text.

### 2. Decision extraction
Each sentence is checked against a list of decision-indicating regex patterns (e.g. `we decided`, `agreed to`, `let's go with`, `the decision is`, `final decision`, `consensus is`). Matching sentences are returned with their speaker.

### 3. Action item extraction
Each sentence is checked, in order, against several rule-based patterns:

1. `Action item[s] [for Name]: ...` — explicit markers
2. `Name will <task>` — third-person assignment
3. `Name to <task>` — explicit assignment (e.g. "Alice to send the deck")
4. `Name, can/could you <task>` — direct address
5. `I'll <task>` / `I will <task>` — first-person self-commitment (assignee = the speaker of that line)
6. A bare imperative verb at the start of the sentence (`Send`, `Follow up`, `Schedule`, `Review`, ...) — assignee falls back to a capitalized name mentioned in the sentence, or the speaker

A small blocklist (`NON_NAME_WORDS`) prevents common sentence-initial words like "Glad", "Sure", or "Nothing" from being mistaken for a person's name in the "X will/to ..." patterns.

Each matched action item is then scanned with `extract_due_date`, which looks for `by <phrase>`, `due <phrase>`, `before <phrase>`, or bare phrases like `next week` / `end of month`.

### 4. Extractive summary (TextRank-style, from scratch)
1. Tokenize every sentence, dropping a small built-in stopword list.
2. Build a sentence-similarity graph: edge weight between two sentences = shared distinct words, normalized by `log(len(A)) + log(len(B))` (the standard TextRank sentence-similarity formula).
3. Run power-iteration PageRank (damping = 0.85) over that graph by hand — no `networkx`.
4. Take the top-N sentences by centrality score, then re-order them by their original position in the transcript so the summary reads coherently.

## Usage

```bash
python summarize.py samples/product_launch.txt
```

```
=== Meeting Summary: samples/product_launch.txt ===

-- Key Decisions --
  - [Alice] We decided last time to launch on October 15th, and I want to confirm that's still the plan.
  - [Alice] Let's go with October 15th as the official launch date then.

-- Action Items --
  - [Bob] I'll set up the staging environment and run the load tests before the launch.
  - [Carol] (due: Friday) Carol, can you finalize the pricing page mockups by Friday?
  - [Dave] (due: next Wednesday) Dave will update the marketing site copy by next Wednesday.
  - [Alice] (due: tomorrow) I'll reach out to the support team lead by tomorrow.

-- Summary --
We still need final sign-off on the new pricing page mockups. We decided last
time to launch on October 15th, and I want to confirm that's still the plan.
Agreed, October 15th works for engineering as long as QA starts by next week.
Let's go with October 15th as the official launch date then.
```

Optional flag: `--sentences N` to control the target summary length (default `4`).

### Sample transcripts

Three self-authored transcripts are bundled in [`samples/`](samples/):

- `product_launch.txt` — a product-launch planning meeting
- `budget_review.txt` — a quarterly marketing budget review
- `incident_retro.txt` — an engineering incident retrospective

## Project layout

```
meeting-summarizer/
├── summarize.py              # CLI entry point
├── summarizer/
│   ├── __init__.py
│   └── core.py                # parsing, extraction, and summarization logic
├── samples/                   # bundled example transcripts
├── tests/
│   └── test_summarizer.py     # unit tests
├── .github/workflows/tests.yml
├── LICENSE
└── README.md
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

Tests cover: decision extraction on known phrasing, action-item extraction with correct assignee (third-person, first-person, and direct-address forms), due-date phrase extraction, extractive-summary centrality on a crafted transcript with a known "hub" sentence, empty and very short transcripts, and transcripts with no decisions or action items (must return empty lists, not errors).

Continuous integration ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the same suite on Python 3.11 and 3.12 on every push and pull request.

## Limitations

This is a rule-based, extractive tool — it has no semantic understanding. It will miss decisions/action items phrased in ways its patterns don't cover, and it can occasionally over- or under-attribute an assignee. It's intended as a fast, transparent, fully local first pass over a transcript, not a substitute for a careful read.

## License

MIT — see [LICENSE](LICENSE).
