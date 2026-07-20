# ERA V5 · Assignment 2 — Lossless India BPE Tokenizer (resubmission 2)

**English · Hindi · Telugu · Sanskrit** — one shared **10,000-token** BPE tokenizer,
**lossless** (passes the round-trip gate) and evaluated on the **faithful Wikipedia
Markdown** corpus.

## Why this version exists — the round-trip gate

The grader added a hard gate:

> `decode(encode(text))` must preserve the same **visible non-whitespace characters**.

The previous submission used a normalizer that replaced every `[^\p{L}\p{M}\p{N}]+`
run with a space, which **deleted punctuation**:

```text
"India's population is 1,428,627,663."  ->  "India s population is 1 428 6 27 6 63"
```

That scored **0** on the gate. This tokenizer is **lossless** and now round-trips it exactly.

## Results (faithful Wikipedia Markdown corpus)

**Round-trip gate: PASSED** — verified on every corpus and on adversarial samples
(punctuation, URLs, Markdown tables, Chinese/Tamil/emoji via byte-fallback), using the
default `decode(...)` (i.e. `skip_special_tokens=True`, which is what the grader uses).

| Language | tokens | word-ish units | fertility | note |
|----------|-------:|---------------:|----------:|------|
| **Hindi**  | 50,068  | 42,151 | **1.1878** | ✅ &lt; 1.2 |
| Telugu     | 21,997  | 16,109 | 1.3655 | |
| Sanskrit   | 9,547   | 6,720  | 1.4207 | 4th language |
| English    | 130,178 | 89,417 | 1.4559 | |

```text
min fertility = 1.1878  (Hindi)   -> rule "at least one language < 1.2" satisfied
max fertility = 1.4559  (English)
spread        = 1.4559 − 1.1878 = 0.2680
raw score     = 1000 / 0.2680    = 3731.0
```

- **Vocabulary** = exactly 10,000 · **round-trip gate PASSED** · **at least one language (Hindi) &lt; 1.2**.

## How it is lossless

A punctuation-stripping normalizer gives low fertility but is lossy. A GPT-2 style
**ByteLevel** tokenizer is lossless but byte-explodes Devanagari/Telugu (each char = 2–3
bytes), pushing Hindi/Telugu fertility to ~3.3–3.7. This tokenizer uses the middle path:

- **Model:** BPE, `byte_fallback=True`, all 256 `<0xNN>` byte tokens present (as
  **non-special** tokens, so `decode()` never drops them).
- **Pre-tokenizer:** `Metaspace` (character-level — Indic scripts stay 1 unit/char, not
  2–3 bytes), `prepend_scheme="always"`, `split=True`.
- **Decoder:** `ByteFallback → Metaspace → Fuse`.
- **No lossy normalizer.**

Result: `decode(encode(x))` reconstructs the visible text for **any** input (byte-fallback
covers characters outside the four training scripts), while Indic fertility stays ~1.2–1.4.

## The core trade-off (why the score is ~3,700, not ~13,000)

Under lossless tokenization, Sanskrit and Telugu (dense compounds, small pages) floor near
~1.36–1.42, while English compresses toward ~1.2. If all four are allowed to cluster
(spread ~0.08, score ~13,000) **no** language dips below 1.2. Forcing **at least one
language < 1.2** (the assignment rule) requires over-weighting one language (Hindi here),
which widens the spread to ~0.27. `3,731` is the best raw score that still satisfies the
rule — see `sweep_lossless.py` for the full search.

## Corpus extraction (exact, reproducible)

`build_wiki_faithful_markdown.py` fetches Wikipedia REST HTML and converts it to faithful
Markdown (keeps links, tables, references, image links, navboxes, categories; strips only
`script`/`style`/`meta`/`link` machinery). Same method as the instructor's reference — for
English, Hindi and Telugu the word-ish counts (89,417 / 42,151 / 16,109) are identical.

| code | Wikipedia | page title | word-ish units |
|---|---|---|---:|
| `en` | en.wikipedia.org | India     | 89,417 |
| `hi` | hi.wikipedia.org | भारत      | 42,151 |
| `te` | te.wikipedia.org | భారతదేశం  | 16,109 |
| `sa` | sa.wikipedia.org | भारतम्    | 6,720  |

## Training

`train_tokenizer.py` — HuggingFace `tokenizers`:

- vocab size **10,000**, `min_frequency=1`
- Metaspace pre-tokenizer + byte-fallback BPE + `ByteFallback→Metaspace→Fuse` decoder (above)
- per-language training weights **`{en:2, hi:6, te:9, sa:10}`** (tuned so Hindi &lt; 1.2 and
  the spread is minimised — see `sweep_lossless.py`)
- post-step: mark the 256 `<0xNN>` byte tokens **non-special** so `decode()` is lossless.

## Reproduce

```bash
pip install tokenizers regex requests beautifulsoup4 lxml markdownify

python build_wiki_faithful_markdown.py   # fetch corpus/*.faithful.txt  (needs internet)
python train_tokenizer.py                # -> tokenizer.json + metrics.json
python evaluate_tokenizer.py             # prints GATE result + fertilities + score
```

`evaluate_tokenizer.py` loads **only** `tokenizer.json` + `corpus/*.faithful.txt`, checks
the round-trip gate, and recomputes every number — reproducible from the shipped tokenizer.

## Files

- **`tokenizer.json`** — the tokenizer (standard HuggingFace format; kept **separate for download**).
- `build_wiki_faithful_markdown.py` — fetch + convert Wikipedia → faithful Markdown corpus.
- `train_tokenizer.py` — train the lossless 10k BPE.
- `evaluate_tokenizer.py` — check the gate + reproduce fertilities/score from `tokenizer.json`.
- `sweep_lossless.py` — the weight search behind `{en:2, hi:6, te:9, sa:10}`.
- `metrics.json` — machine-readable results (incl. gate flags).
- `tokens.txt` — the full 10,000-token vocabulary list.
- `corpus/*.faithful.md` / `*.faithful.txt` / `*.meta.json` — exact corpus snapshots.
- `index.html` — self-contained widget: **download** `tokenizer.json`, **live round-trip gate**
  demo, **live** fertility recompute (faithful JS re-implementation of Metaspace → BPE →
  byte-fallback, verified to match Python token-for-token), a tokenization playground, and a
  searchable, paginated browser over all 10,000 tokens.

## The widget

Open `index.html` in any browser (self-contained). Type into the round-trip box to watch
`decode(encode(x))` preserve your text; click **“Re-compute fertilities live”** to reproduce
the table in-browser; **“Download tokenizer.json”** to grab the file. Host via Netlify Drop
(drag `index.html` in) or GitHub Pages.
