# ERA V5 Assignment 2 Reference Solution

This folder contains a reproducible reference solution for the multilingual BPE tokenizer assignment.

The goal was to build one shared tokenizer with a 10,000-token vocabulary for India Wikipedia pages in:

- English
- Hindi
- Telugu
- Maithili

The evaluation corpus here uses a **wiki-faithful Markdown** conversion rather than clipped article prose. Links, URLs, tables, references, image links, navboxes, and categories are preserved where the HTML-to-Markdown conversion emits them.

## Result

| Language | Tokens | Word-ish Units | Fertility |
|---|---:|---:|---:|
| English | 109,210 | 89,417 | 1.221356 |
| Hindi | 50,256 | 42,151 | 1.192285 |
| Telugu | 21,735 | 16,109 | 1.349246 |
| Maithili | 3,509 | 2,568 | 1.366433 |

```text
Spread = 1.366433 - 1.192285 = 0.174148
Raw score = 1000 / 0.174148 = 5742.24
```

Hindi satisfies the required threshold:

```text
Hindi fertility = 1.192285 <= 1.2
```

English is slightly above 1.2:

```text
English fertility = 1.221356
```

So treat this as a strong reference implementation and a reproducibility baseline, not a claim that every constraint is perfectly optimized.

## Folder Contents

```text
tokenizer.json                    trained tokenizer
metrics.json                      saved metrics for this tokenizer
build_wiki_faithful_markdown.py   fetch + convert Wikipedia pages
train_tokenizer.py                train tokenizer from corpus
evaluate_tokenizer.py             evaluate tokenizer.json
corpus/*.faithful.md              generated Markdown corpus snapshots
corpus/*.faithful.txt             same corpus as plain text input
corpus/*.meta.json                corpus metadata
```

## Setup

Install dependencies:

```bash
pip install tokenizers regex requests beautifulsoup4 lxml markdownify
```

## Rebuild the Corpus

```bash
python build_wiki_faithful_markdown.py
```

This fetches Wikipedia REST HTML and writes:

```text
corpus/en.faithful.md
corpus/hi.faithful.md
corpus/te.faithful.md
corpus/mai.faithful.md
```

The corpus counts may change if Wikipedia pages change. The included corpus snapshots are the ones used to produce the metrics above.

## Train the Tokenizer

```bash
python train_tokenizer.py
```

Training choices:

- Model: HuggingFace BPE
- Vocab size: 10,000
- `min_frequency=1`
- Normalizer:
  - NFKC
  - replace every non-letter/mark/number run with a space
- Pretokenizer: whitespace
- Training weights:

```json
{
  "en": 3,
  "hi": 4,
  "te": 4,
  "mai": 2
}
```

The non-letter replacement is deliberate. The faithful Markdown corpus contains a lot of punctuation, URLs, pipes, brackets, and Markdown syntax. The assignment’s fertility denominator is word-ish units, so the tokenizer is also trained to focus its budget on letter/mark/number units rather than wasting merges on punctuation.

## Evaluate

```bash
python evaluate_tokenizer.py
```

The score formula is:

```text
fertility(language) = token_count(language) / wordish_unit_count(language)
score = 1000 / (max_fertility - min_fertility)
```

The evaluator also prints a Hindi penalty score:

```text
hindi_penalty = exp(max(0, hindi_fertility / 1.2 - 1))
hindi_adjusted_score = raw_score / hindi_penalty
```

Since this tokenizer has Hindi below 1.2, its Hindi penalty factor is 1.

## Notes for Students

Do not report numbers from a clipped page or a hidden private corpus. The important part of this assignment is reproducibility.

A good submission should include:

- the exact tokenizer file
- code or clear method used to build it
- the exact Wikipedia corpus extraction process
- token counts for all four languages
- fertility ratios
- raw score calculation
- a live widget or notebook that lets the grader inspect/download the tokenizer

If your tokenizer relies on a custom JSON format, include the encoder code. A vocab list without the actual encoding algorithm is not enough to reproduce your score.
