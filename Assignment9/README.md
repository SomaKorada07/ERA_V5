# Assignment 9 — The Loss Harness (nanoGPT / GPT-2)

Make the next-token training loss **correct and observable**, then add a second output head.
Architecture: Andrej Karpathy's **nanoGPT / GPT-2**. Data: **FineWeb** (`HuggingFaceFW/fineweb`,
`sample-10BT`), streamed so there is no multi-GB download.

Notebook: [`Assignment9_loss_harness.ipynb`](./Assignment9_loss_harness.ipynb) — runs top to bottom
on Apple Silicon (MPS), CUDA, or CPU.

## What it covers

**Part 1 — the harness (seven numbers)**
1. Every tensor shape printed, each dimension named.
2. The shift verified by decoding **strings** (`input i → target i+1`), not ids.
3. Padding masked; the contributing-token count provably drops.
4. Two documents packed into one sequence; the cross-document boundary masked, loss shown before/after.
5. Perplexity of an untrained model ≈ vocab size (≈ 50,257). *This check caught a real init bug.*
6. Tied vs untied head parameter counts (difference = `vocab × n_embd`).
7. Peak memory: ordinary cross-entropy vs a self-written chunked version, with the ratio.

**Part 2 — a second head** predicting token `t+2`. Both losses reported separately plus their sum;
the `t+2` head's loss stays above the `t+1` head's and the gap widens over training.

**Part 3 — the off-by-one trap.** A deliberately reversed shift produces a beautiful loss curve and
raises no exception; printing the strings exposes it. *Always print the strings.*

## Run it on a Mac

Requires Python 3.10+. From this folder:

```bash
# 1. (recommended) create a virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install torch tiktoken datasets matplotlib jupyter

# 3a. open interactively and Run All
jupyter notebook Assignment9_loss_harness.ipynb

# 3b. or run headless, top to bottom
pip install nbconvert ipykernel
python3 -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 Assignment9_loss_harness.ipynb
```

Notes:
- On Apple Silicon the notebook auto-selects the **MPS** backend. Full run (incl. ~250 training
  steps) is a few minutes.
- FineWeb is **streamed**; only ~80k tokens are pulled. If the Hugging Face Hub is unreachable, the
  notebook falls back to a small built-in corpus so it still runs end to end.
- Optionally set a token to avoid Hub rate limits: `export HF_TOKEN=...`.

## On Google Colab
Upload the `.ipynb`, then **Runtime → Run all**. The first cell (`%pip install ...`) handles deps.
