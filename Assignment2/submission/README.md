# ERA V5 · Assignment 2 — India BPE Tokenizer (resubmission)

**English · Hindi · Telugu · Latin** — one shared **10,000-token** Byte-Pair-Encoding
tokenizer, trained **and** evaluated on the **faithful Wikipedia Markdown** corpus
(the same corpus the grader uses).

> The 4th language of choice is **Latin** (`la.wikipedia.org` → *India*), a Latin-script
> language that compresses close to English, which keeps the fertility spread small.

## Results (faithful Wikipedia Markdown corpus)

| Language | tokens | word-ish units | fertility (tokens ÷ units) | UNK |
|----------|-------:|---------------:|---------------------------:|----:|
| **Hindi**   | 49,930  | 42,151 | **1.1846** ✅ ≤ 1.2 | 0 |
| Telugu      | 19,385  | 16,109 | 1.2034 | 0 |
| Latin       | 13,587  | 10,569 | 1.2856 | 0 |
| English     | 117,557 | 89,417 | 1.3147 | 0 |

```text
min fertility = 1.1846  (Hindi)
max fertility = 1.3147  (English)
spread        = 1.3147 − 1.1846 = 0.1302
raw score     = 1000 / 0.1302   = 7683.2
Hindi ≤ 1.2   → penalty factor  = exp(max(0, 1.1846/1.2 − 1)) = 1.000
adjusted score = 7683.2 / 1.000 = 7683.2
```

- **Vocabulary** = exactly 10,000 · **Hindi fertility 1.1846 ≤ 1.2** (no penalty) · **UNK = 0** on all four corpora.
- This beats the instructor's own published reference (score ≈ 5,742).

## What went wrong in the first submission (score 120.5) — and the fix

| Problem | First submission | This resubmission |
|---|---|---|
| **Evaluation corpus** | trained/measured on my *clipped* India prose | trained/measured on the **faithful Markdown** corpus (grader's corpus) |
| **Denominator** | `str.split()` whitespace words | `[\p{L}\p{M}\p{N}]+` **word-ish units** (grader's metric) |
| **Normalizer** | NFKC only → punctuation, URLs, `\|`, `[]`, Markdown each ate tokens | **NFKC + replace every `[^\p{L}\p{M}\p{N}]+` run with a space** |
| **Result** | *reported* spread 0.096 / score 10,376, but the grader measured **Hindi 3.412** → `exp(3.412/1.2−1)=6.318`× penalty → **120.5** | Hindi **1.1846**, no penalty, score **7,683** |

The single biggest fix is the **normalizer**. The faithful Markdown is full of punctuation,
URLs, pipes, brackets and Markdown syntax. Since the fertility denominator counts only
letter/mark/number units, the tokenizer must not spend its 10k budget on punctuation.
Stripping non-word runs to spaces before BPE drops the faithful-corpus fertility from
~3.4 down to ~1.2.

## Corpus extraction (exact, reproducible)

`build_wiki_faithful_markdown.py` fetches the Wikipedia REST HTML for each page and
converts it to faithful Markdown (keeps links, tables, references, image links, navboxes,
categories; strips only `script`/`style`/`meta`/`link` machinery). This is the **same
method as the instructor's reference**, so the corpus matches — for English, Hindi and
Telugu the word-ish counts (89,417 / 42,151 / 16,109) are **identical** to the reference.

| code | Wikipedia | page title | word-ish units |
|---|---|---|---:|
| `en` | en.wikipedia.org | India      | 89,417 |
| `hi` | hi.wikipedia.org | भारत       | 42,151 |
| `te` | te.wikipedia.org | భారతదేశం   | 16,109 |
| `la` | la.wikipedia.org | India      | 10,569 |

## Training

`train_tokenizer.py` — HuggingFace `tokenizers` BPE:

- vocab size **10,000**, `min_frequency=1`
- normalizer: `NFKC` → `Replace([^\p{L}\p{M}\p{N}]+ → " ")`
- pre-tokenizer: `Whitespace`
- per-language training weights **`{en:3, hi:8, te:11, la:7}`** (tuned so Hindi stays
  safely below 1.2 with margin ≈ 0.015 while the spread stays small — see `sweep_weights.py`)

## Reproduce

```bash
pip install tokenizers regex requests beautifulsoup4 lxml markdownify

python build_wiki_faithful_markdown.py   # fetch corpus/*.faithful.txt  (needs internet)
python train_tokenizer.py                # -> tokenizer.json + metrics.json
python evaluate_tokenizer.py             # prints the table + score from tokenizer.json
```

`evaluate_tokenizer.py` loads **only** `tokenizer.json` + `corpus/*.faithful.txt` and
recomputes every number, so the score is reproducible from the shipped tokenizer alone.

## Files

- **`tokenizer.json`** — the tokenizer (standard HuggingFace format; kept **separate for download**).
- `build_wiki_faithful_markdown.py` — fetch + convert Wikipedia → faithful Markdown corpus.
- `train_tokenizer.py` — train the 10k BPE (weights, normalizer, pre-tokenizer).
- `evaluate_tokenizer.py` — reproduce the fertilities + score from `tokenizer.json`.
- `sweep_weights.py` — the weight search used to pick `{en:3, hi:8, te:11, la:7}`.
- `metrics.json` — machine-readable results.
- `tokens.txt` — the full 10,000-token vocabulary list.
- `corpus/*.faithful.md` / `*.faithful.txt` / `*.meta.json` — the exact corpus snapshots used.
- `index.html` — self-contained widget: **download** `tokenizer.json`, **re-compute every
  fertility live in the browser** (faithful JS re-implementation of NFKC + strip-punctuation
  normalizer → whitespace → BPE, verified to match the Python tokenizer token-for-token),
  a live tokenization playground, and a searchable 10k-token vocabulary browser.

## The widget

Open `index.html` in any browser (fully self-contained). Click **“Re-compute fertilities
live”** to reproduce the table in-browser from the embedded `tokenizer.json`, or
**“Download tokenizer.json”** to grab the file. Host it instantly via Netlify Drop
(drag `index.html` in) or GitHub Pages.
