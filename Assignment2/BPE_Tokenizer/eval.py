#!/usr/bin/env python3
"""
Reproduce the tokenizer's fertility scores.

    pip install tokenizers
    python eval.py

Loads tokenizer.json and each corpus_<lang>.txt (from the current directory or ./out/)
and computes, per language,  X = (BPE tokens) / (whitespace words)  over that language's
whole India-article corpus. Prints X1..X4, the score 1000/(X4-X1), the English<=1.2 check,
and the <unk> count (must be 0). Deterministic -> reproduces exactly.

To evaluate the COMPLETE India pages in all four languages, first run:
    python fetch_corpora.py     # downloads the 4 full articles -> data/*.txt
    python train_final.py       # retrains on the full pages -> out/*
    python eval.py              # reports fertilities for the complete pages
"""
import os
from tokenizers import Tokenizer

LANGS = [("en","English"), ("hi","Hindi"), ("te","Telugu"), ("la","Latin")]

def find(name):
    for base in ("out", "."):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name + " (looked in ./out and .)")

tok = Tokenizer.from_file(find("tokenizer.json"))
UNK = "<unk>"
rows, total_unk = [], 0
for code, name in LANGS:
    text   = open(find(f"corpus_{code}.txt"), encoding="utf-8").read()
    words  = len(text.split())            # denominator: whitespace words
    enc    = tok.encode(text)
    tokens = len(enc.ids)                  # numerator: BPE tokens
    unk    = enc.tokens.count(UNK)         # must be 0
    total_unk += unk
    rows.append((code, name, tokens, words, tokens/words, unk))

en_X = next(r[4] for r in rows if r[0] == "en")
rows_sorted = sorted(rows, key=lambda r: r[4])
X1, X4 = rows_sorted[0][4], rows_sorted[-1][4]
spread = X4 - X1
score = 1000/spread if spread else float("inf")

print(f"realized vocabulary size: {tok.get_vocab_size()}  (cap 10000)\n")
print(f"{'lang':8s}{'tokens':>9s}{'words':>8s}{'X = tok/word':>15s}{'UNK':>6s}")
for code, name, t, w, x, u in rows_sorted:
    print(f"{name:8s}{t:>9d}{w:>8d}{x:>15.4f}{u:>6d}")
print()
print(f"X1 (min) = {X1:.4f}  [{rows_sorted[0][1]}]")
print(f"X4 (max) = {X4:.4f}  [{rows_sorted[-1][1]}]")
print(f"spread X4 - X1 = {spread:.4f}")
print(f"SCORE = 1000 / (X4 - X1) = {score:.1f}")
print(f"English = {en_X:.4f}  <= 1.2 : {en_X <= 1.2}")
print(f"total <unk> tokens : {total_unk}   (must be 0)")
