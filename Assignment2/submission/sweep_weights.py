#!/usr/bin/env python3
"""Sweep training weights to minimize spread while keeping Hindi <= 1.2.

Metric matches the grader exactly:
    wordish_units(text) = number of [\\p{L}\\p{M}\\p{N}]+ runs
    fertility(lang)     = tokens(lang) / wordish_units(lang)
    spread              = max(fertility) - min(fertility)
    raw_score           = 1000 / spread
    hindi_penalty       = exp(max(0, hindi_fertility/1.2 - 1))
    adjusted_score      = raw_score / hindi_penalty
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import regex
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Replace, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "la"]


def wordish_units(text: str) -> int:
    return len(regex.findall(r"[\p{L}\p{M}\p{N}]+", text))


def make_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC(), Replace(Regex(r"[^\p{L}\p{M}\p{N}]+"), " ")])
    tok.pre_tokenizer = Whitespace()
    return tok


def train_eval(weights: dict[str, int]) -> dict:
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
    units = {c: wordish_units(t) for c, t in texts.items()}
    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        for c, t in texts.items():
            p = Path(tmp) / f"{c}.txt"
            p.write_text(t, encoding="utf-8")
            files.extend([str(p)] * weights[c])
        tok = make_tokenizer()
        tok.train(files, BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]"]))
    tc = {c: len(tok.encode(texts[c]).ids) for c in LANGS}
    fert = {c: tc[c] / units[c] for c in LANGS}
    spread = max(fert.values()) - min(fert.values())
    raw = 1000 / spread
    pen = math.exp(max(0.0, fert["hi"] / 1.2 - 1.0))
    return {"weights": weights, "fert": fert, "spread": spread, "raw": raw,
            "penalty": pen, "adj": raw / pen, "vocab": tok.get_vocab_size()}


CANDIDATES = [
    {"en": 3, "hi": 8, "te": 11, "la": 7},
    {"en": 3, "hi": 8, "te": 9, "la": 7},
    {"en": 3, "hi": 8, "te": 8, "la": 7},
    {"en": 3, "hi": 8, "te": 10, "la": 8},
    {"en": 4, "hi": 9, "te": 10, "la": 8},
    {"en": 3, "hi": 8, "te": 9, "la": 8},
    {"en": 4, "hi": 8, "te": 9, "la": 7},
    {"en": 4, "hi": 8, "te": 9, "la": 8},
    {"en": 4, "hi": 8, "te": 10, "la": 8},
    {"en": 3, "hi": 7, "te": 9, "la": 7},
]


def main() -> int:
    best = None
    for w in CANDIDATES:
        r = train_eval(w)
        f = r["fert"]
        flag = "OK " if f["hi"] <= 1.2 else "PEN"
        print(f"{flag} w={w} | en={f['en']:.3f} hi={f['hi']:.3f} te={f['te']:.3f} la={f['la']:.3f} "
              f"| spread={r['spread']:.4f} raw={r['raw']:.0f} adj={r['adj']:.0f}")
        # prefer valid (hi<=1.2) with highest adjusted score
        key = (f["hi"] <= 1.2, r["adj"])
        if best is None or key > best[0]:
            best = (key, r)
    print("\nBEST:", best[1]["weights"], "adj=", round(best[1]["adj"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
