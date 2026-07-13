#!/usr/bin/env python3
"""
Evaluate tokenizer.json on the faithful Wikipedia Markdown corpus.

Reproduces the grader's numbers exactly:

    wordish_units(text) = number of [\\p{L}\\p{M}\\p{N}]+ runs
    fertility(lang)     = tokens(lang) / wordish_units(lang)
    spread              = max(fertility) - min(fertility)
    raw_score           = 1000 / spread
    hindi_penalty       = exp(max(0, hindi_fertility / 1.2 - 1))
    adjusted_score      = raw_score / hindi_penalty

Run:
    pip install tokenizers regex
    python evaluate_tokenizer.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS = ["en", "hi", "te", "la"]
NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "la": "Latin"}


def wordish_units(text: str) -> int:
    return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))


def main() -> int:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = wordish_units(text)
        enc = tokenizer.encode(text)
        tokens = len(enc.ids)
        rows[code] = {
            "name": NAMES[code],
            "tokens": tokens,
            "wordish_units": units,
            "ratio": tokens / units,
            "unk": enc.tokens.count("[UNK]"),
        }

    ratios = {c: rows[c]["ratio"] for c in LANGS}
    lo = min(ratios, key=ratios.get)
    hi = max(ratios, key=ratios.get)
    spread = ratios[hi] - ratios[lo]
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, ratios["hi"] / 1.2 - 1.0))

    result = {
        "vocab_size": tokenizer.get_vocab_size(),
        "rows": rows,
        "min_lang": NAMES[lo],
        "max_lang": NAMES[hi],
        "spread": spread,
        "raw_score": score,
        "hindi_fertility": ratios["hi"],
        "hindi_ok_leq_1.2": ratios["hi"] <= 1.2,
        "hindi_penalty_factor": hindi_penalty,
        "adjusted_score": score / hindi_penalty,
    }

    print(f"{'language':10s}{'tokens':>9s}{'word-ish':>10s}{'fertility':>11s}{'UNK':>6s}")
    for c in sorted(LANGS, key=lambda x: ratios[x]):
        r = rows[c]
        print(f"{r['name']:10s}{r['tokens']:>9d}{r['wordish_units']:>10d}{r['ratio']:>11.4f}{r['unk']:>6d}")
    print()
    print(f"min fertility : {ratios[lo]:.4f} [{NAMES[lo]}]")
    print(f"max fertility : {ratios[hi]:.4f} [{NAMES[hi]}]")
    print(f"spread        : {spread:.4f}")
    print(f"raw score     : {score:.1f}   = 1000 / spread")
    print(f"Hindi <= 1.2  : {ratios['hi']:.4f}  -> {ratios['hi'] <= 1.2}  (penalty x{hindi_penalty:.3f})")
    print(f"adjusted score: {score / hindi_penalty:.1f}")
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
