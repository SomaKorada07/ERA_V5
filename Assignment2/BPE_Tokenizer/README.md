# India BPE Tokenizer — English · Hindi · Telugu · Latin

A single **10,000-token** Byte-Pair-Encoding tokenizer trained on the **complete** *India*
Wikipedia article in four languages (4th language of choice: **Latin**, from la.wikipedia.org).

## Metric & score

- Fertility per language: **X = BPE tokens ÷ whitespace-words** (Python `str.split`),
  measured on that language's **entire** India article.
- Score = **1000 ÷ (X_max − X_min)**.

## Results (full pages, all four languages)

| Language | tokens | words | X = tokens ÷ words | UNK |
|----------|-------:|------:|-------------------:|----:|
| English (X₁) | 8,388 | 7,064 | **1.1874** | 0 |
| Hindi | 4,981 | 3,953 | 1.2601 | 0 |
| Telugu | 3,034 | 2,391 | 1.2689 | 0 |
| Latin (X₄) | 864 | 673 | **1.2838** | 0 |

- spread = X₄ − X₁ = **0.0964**
- **score = 1000 / 0.0964 ≈ 10,376**
- English ≤ 1.2 ✓ (1.1874, margin 0.0126), vocabulary = exactly 10,000, total `<unk>` = 0

## Why Latin scores so much higher than Tamil did

With Tamil as the 4th language the score was ~1,872: Tamil is an agglutinative Dravidian
script whose fertility floors near 1.7, far above English's ~1.2, so the spread stayed wide.
**Latin uses the Latin alphabet**, like English, so its subwords compress efficiently and its
fertility (1.284) lands right next to English/Hindi/Telugu. All four fertilities fit inside a
0.096 band, which is what drives the 5.5× higher score.

## Three hard requirements, all satisfied

1. **Vocabulary ≤ 10,000** — realized exactly 10,000 (9,744 BPE merges + 256 byte tokens),
   asserted in `train_final.py`, verified on the loaded `tokenizer.json`.
2. **English ≤ 1.2** — 1.1874 (margin 0.0126).
3. **Zero `<unk>`** — character-level base + `byte_fallback=true` (all 256 byte tokens); any
   input encodes to byte tokens instead of `<unk>`. The playground demonstrates this live.

## Honesty note on the score (important — you said you'll run it yourselves)

- The **in-corpus** score (tokenizing the same full India pages) is **10,376**, and `eval.py`
  reproduces it exactly from `tokenizer.json` + the shipped `corpus_*.txt`.
- On a **held-out** split (train on 80% of the pages, evaluate on the unseen 20%) the spread
  widens and the equivalent score is ~1,700. That gap is expected: with a 10k vocabulary and
  full-page training, the tokenizer partly memorizes these specific pages. If you tokenize the
  **same** full pages, you'll see ~10,376; if your cleaned page text differs (newer revision,
  different cleaning), the number will be lower.
- The English margin is thin (0.0126). If your English cleaning yields slightly more
  tokens/word, English could edge over 1.2. For a safer ~0.08 margin at a lower score (~4,500),
  retrain with weights `{"en":2.0,"hi":1.3,"te":2.0,"la":1.6}` in `train_final.py`.

## Files

- `index.html` — the widget. Self-contained; embeds the tokenizer and **recomputes every number
  live in your browser** (verified to match `eval.py` token-for-token). Includes a
  **Tokenization Playground** (type anything → live token count, words, fertility, per-token
  colouring by script, red byte-fallback chips, zero-`<unk>` proof), the fertility bars, the
  stats/score table, and a searchable 10,000-token vocabulary with downloads.
- `tokenizer.json` — the tokenizer (loads directly in HuggingFace `tokenizers`).
- `tokens.txt` / `tokens.json` — the full 10,000-token list.
- `corpus_{en,hi,te,la}.txt` — the exact full-page corpora used (also under `data/`).
- `train_final.py` — training + balancing + byte fallback + 10k assertion (weights
  `{"en":1.8,"hi":1.5,"te":2.0,"la":2.0}`).
- `eval.py` — reproduces the score from `tokenizer.json` + `corpus_*.txt`.
- `result.json` — machine-readable results.

## Reproduce

```bash
pip install tokenizers
python eval.py          # prints per-language X, spread, score, English<=1.2 check, UNK=0
# or retrain from scratch on the full pages:
python train_final.py   # reads data/*.txt -> writes out_la/*
```

## Host the widget (get a URL)

`index.html` is fully self-contained (only external calls are web fonts).

- **Netlify Drop** — https://app.netlify.com/drop and drag `index.html` in → instant public URL.
- **GitHub Pages** — commit `index.html`, enable Pages.
- **Local** — open `index.html` in any browser.
